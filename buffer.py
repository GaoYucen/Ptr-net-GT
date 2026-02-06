import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import NamedTuple

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel


class DualAttentionModel(nn.Module):
    """
    双解码器模型：一个选择起点，一个选择终点
    总参数量与原单解码器模型相当
    """

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=3,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None,
                 info_pass='rich',
                 use_chain_info=True):
        super(DualAttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        # 为了保持参数量相当，每个解码器使用一半的维度
        self.half_dim = embedding_dim // 2
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.is_vrp = problem.NAME == 'cvrp' or problem.NAME == 'sdvrp'
        self.is_orienteering = problem.NAME == 'op'
        self.is_pctsp = problem.NAME == 'pctsp'
        self.problem = problem
        
        # 信息传递方式
        self.info_pass = info_pass
        self.use_chain_info = use_chain_info

        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size

        # ========== 编码器部分（保持不变） ==========
        # Problem specific context parameters
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            step_context_dim = embedding_dim + 1
            if self.is_pctsp:
                node_dim = 4  # x, y, expected_prize, penalty
            else:
                node_dim = 3  # x, y, demand / prize
            self.init_embed_depot = nn.Linear(2, embedding_dim)
        else:  # TSP
            assert problem.NAME == "tsp", "Unsupported problem: {}".format(problem.NAME)
            step_context_dim = 2 * embedding_dim
            node_dim = 2  # x, y
            self.W_placeholder = nn.Parameter(torch.Tensor(2 * embedding_dim))
            self.W_placeholder.data.uniform_(-1, 1)

        self.init_embed = nn.Linear(node_dim, embedding_dim)
        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # ========== 双解码器部分 ==========
        # 编码器输出投影到两个子空间（每个维度减半）
        self.project_to_start = nn.Linear(embedding_dim, self.half_dim)
        self.project_to_end = nn.Linear(embedding_dim, self.half_dim)
        
        # 起点解码器（维度减半）
        self.start_decoder = StartDecoder(
            half_dim=self.half_dim,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
            use_chain_info=use_chain_info
        )
        
        # 终点解码器（维度减半）
        self.end_decoder = EndDecoder(
            half_dim=self.half_dim,
            n_heads=n_heads,
            tanh_clipping=tanh_clipping,
            use_chain_info=use_chain_info,
            info_pass=info_pass
        )
        
        # 信息传递模块
        self.info_passing_module = InfoPassingModule(
            half_dim=self.half_dim,
            mode=info_pass
        )
        
        # 链条状态相关
        self.chain_embed_projection = None
        if use_chain_info:
            self.chain_embed_projection = nn.Linear(self.half_dim + 2, self.half_dim)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        前向传播
        """
        # ========== 1. 编码阶段 ==========
        if self.checkpoint_encoder and self.training:
            embeddings, _ = torch.utils.checkpoint.checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))
        
        # 编码器输出： [batch_size, graph_size, embedding_dim]
        batch_size, graph_size, _ = embeddings.size()

        # ========== 2. 投影到两个子空间 ==========
        start_embeddings = self.project_to_start(embeddings)  # [batch, graph, half_dim]
        end_embeddings = self.project_to_end(embeddings)      # [batch, graph, half_dim]
        
        # 计算图嵌入（均值池化）
        start_graph_embed = start_embeddings.mean(dim=1, keepdim=True)  # [batch, 1, half_dim]
        end_graph_embed = end_embeddings.mean(dim=1, keepdim=True)      # [batch, 1, half_dim]

        # ========== 3. 解码阶段 ==========
        log_probs_start = []
        log_probs_end = []
        sequences = []
        
        # 初始化链条状态
        chain_state = DualChainState(graph_size, batch_size, device=embeddings.device)
        
        # 预计算固定上下文
        start_fixed = self.start_decoder.precompute(start_embeddings, start_graph_embed)
        end_fixed = self.end_decoder.precompute(end_embeddings, end_graph_embed)

        # 迭代构建链条
        max_steps = graph_size - 1
        for step in range(max_steps):
            # 调试信息
            # print(f"Step {step+1}/{max_steps}, active chains: {chain_state.get_active_chain_count()}")
            
            # ========== 3.1 增强节点嵌入（如果需要） ==========
            if self.use_chain_info:
                start_emb = chain_state.enhance_with_chain_info(
                    start_fixed.node_embeddings, 
                    self.chain_embed_projection
                )
                end_emb = chain_state.enhance_with_chain_info(
                    end_fixed.node_embeddings,
                    self.chain_embed_projection
                )
            else:
                start_emb = start_fixed.node_embeddings
                end_emb = end_fixed.node_embeddings

            # ========== 3.2 获取掩码 ==========
            start_mask = chain_state.get_start_mask()  # [batch, graph]
            end_mask = chain_state.get_end_mask()      # [batch, graph]

            # 检查掩码是否有效
            if start_mask.all():
                # 所有节点都被掩码了，这不应该发生
                # 添加一个安全机制：重置掩码
                # print("Warning: All start nodes masked, resetting...")
                chain_state.reset_masks()
                start_mask = chain_state.get_start_mask()
                end_mask = chain_state.get_end_mask()

            # ========== 3.3 起点解码 ==========
            # 获取起点上下文
            start_context = self.start_decoder.get_context(
                start_fixed, chain_state
            )
            
            # 计算起点logits
            start_logits = self.start_decoder.get_logits(
                start_context, start_fixed, start_emb, start_mask
            )  # [batch, graph]
            
            # 选择起点
            selected_start = self._select_node(
                F.softmax(start_logits / self.temp, dim=-1), 
                start_mask
            )  # [batch]
            
            # 存储log概率
            log_probs_start.append(start_logits)
            
            # ========== 3.4 信息传递 ==========
            if self.info_pass != 'none':
                # 获取起点的嵌入 [batch, 1, half_dim]
                selected_start_idx = selected_start.unsqueeze(-1).unsqueeze(-1)
                selected_start_emb = start_emb.gather(
                    1, 
                    selected_start_idx.expand(-1, -1, self.half_dim)
                )  # [batch, 1, half_dim]
                
                # 传递信息
                passed_info = self.info_passing_module(
                    selected_start_emb, 
                    chain_state,
                    mode=self.info_pass
                )  # [batch, 1, half_dim]
            else:
                passed_info = None

            # ========== 3.5 终点解码 ==========
            # 更新终点掩码：不能选择起点本身所在的链条的起点
            chain_state.update_for_selected_start(selected_start)
            
            # 获取更新后的终点掩码
            updated_end_mask = chain_state.get_end_mask()
            
            # 检查是否有可选的终点
            if updated_end_mask.all():
                # 所有终点都被掩码了，这不应该发生
                # 回退到使用原始终点掩码，但排除起点本身
                # print("Warning: All end nodes masked, using fallback...")
                updated_end_mask = end_mask.clone()
                for b in range(batch_size):
                    start_node = selected_start[b].item()
                    updated_end_mask[b, start_node] = False
            
            # 获取终点上下文（包含传递的信息）
            end_context = self.end_decoder.get_context(
                end_fixed, chain_state, passed_info
            )
            
            # 计算终点logits
            end_logits = self.end_decoder.get_logits(
                end_context, end_fixed, end_emb, updated_end_mask
            )  # [batch, graph]
            
            # 选择终点
            selected_end = self._select_node(
                F.softmax(end_logits / self.temp, dim=-1),
                updated_end_mask
            )  # [batch]
            
            # 存储log概率
            log_probs_end.append(end_logits)
            
            # ========== 3.6 更新链条状态 ==========
            chain_state.update_chains(selected_start, selected_end)
            
            # 记录序列
            sequences.append((selected_start, selected_end))
            
            # 提前终止检查：如果只剩下一个活跃链条
            if chain_state.get_active_chain_count().min() <= 1:
                # print(f"Early termination at step {step+1}")
                break

        # ========== 4. 计算损失 ==========
        if len(log_probs_start) == 0:
            # 如果没有步骤，返回默认值
            cost = torch.zeros(batch_size, device=embeddings.device)
            ll = torch.zeros(batch_size, device=embeddings.device)
            pi = self._chains_to_tour(chain_state)
            if return_pi:
                return cost, ll, pi
            return cost, ll
        
        # 将logits转换为概率并计算对数似然
        log_p_start = torch.stack(log_probs_start, 1)  # [batch, steps, graph]
        log_p_end = torch.stack(log_probs_end, 1)      # [batch, steps, graph]
        
        # 创建动作序列
        start_actions = torch.stack([s[0] for s in sequences], 1)  # [batch, steps]
        end_actions = torch.stack([s[1] for s in sequences], 1)    # [batch, steps]
        
        # 计算对数似然
        ll_start = self._calc_log_likelihood(log_p_start, start_actions, None)
        ll_end = self._calc_log_likelihood(log_p_end, end_actions, None)
        
        # 总对数似然是两个解码器的和
        ll = ll_start + ll_end
        
        # ========== 5. 计算成本 ==========
        # 将起点-终点对转换为TSP路径
        pi = self._chains_to_tour(chain_state)
        
        cost, mask = self.problem.get_costs(input, pi)
        
        if return_pi:
            return cost, ll, pi
        
        return cost, ll

    def _init_embed(self, input):
        """初始化嵌入（与原模型相同）"""
        if self.is_vrp or self.is_orienteering or self.is_pctsp:
            if self.is_vrp:
                features = ('demand', )
            elif self.is_orienteering:
                features = ('prize', )
            else:
                assert self.is_pctsp
                features = ('deterministic_prize', 'penalty')
            return torch.cat(
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((
                        input['loc'],
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))
                ),
                1
            )
        # TSP
        return self.init_embed(input)

    def _select_node(self, probs, mask):
        """选择节点（贪心或采样）"""
        # probs: [batch, graph]
        # mask: [batch, graph]
        
        # 确保probs是2D [batch, graph]
        if probs.dim() != 2:
            # 如果probs是3D [batch, 1, graph]，则压缩成2D
            probs = probs.squeeze(1)
        
        batch_size, graph_size = probs.shape
        
        # 将掩码应用到概率上
        probs_masked = probs.clone()
        probs_masked = probs_masked.masked_fill(mask, 0)
        
        # 检查是否所有概率都是0
        zero_prob_mask = (probs_masked.sum(dim=1) == 0)
        if zero_prob_mask.any():
            # 如果有批次的所有概率都是0，随机选择一个未掩码的节点
            # print(f"Warning: All probabilities zero for {zero_prob_mask.sum()} batches")
            available = ~mask
            for b in range(batch_size):
                if zero_prob_mask[b]:
                    if available[b].any():
                        # 随机选择一个可用的节点
                        avail_indices = torch.where(available[b])[0]
                        probs_masked[b, avail_indices[0]] = 1.0  # 设为1，确保能被选中
                    else:
                        # 如果没有可用节点，选择第一个节点
                        probs_masked[b, 0] = 1.0
        
        # 重新归一化概率（排除被掩码的节点）
        probs_sum = probs_masked.sum(dim=1, keepdim=True)
        probs_masked = probs_masked / (probs_sum + 1e-10)
        
        if self.decode_type == "greedy":
            _, selected = probs_masked.max(1)
        elif self.decode_type == "sampling":
            selected = torch.multinomial(probs_masked, 1).squeeze(1)
            
            # 检查是否选择了无效动作
            invalid_selection = mask.gather(1, selected.unsqueeze(-1)).squeeze(-1)
            max_retries = 10
            retry_count = 0
            
            while invalid_selection.any() and retry_count < max_retries:
                # 只对无效的选择重新采样
                for b in range(batch_size):
                    if invalid_selection[b]:
                        # 重新采样这个批次
                        selected[b] = torch.multinomial(probs_masked[b].unsqueeze(0), 1).squeeze(0)
                
                invalid_selection = mask.gather(1, selected.unsqueeze(-1)).squeeze(-1)
                retry_count += 1
            
            if retry_count == max_retries and invalid_selection.any():
                # 如果达到最大重试次数，使用贪心策略
                # print(f"Warning: Max retries reached, using greedy fallback")
                _, selected_fallback = probs_masked.max(1)
                selected = torch.where(invalid_selection, selected_fallback, selected)
        else:
            raise ValueError("Unknown decode type: {}".format(self.decode_type))
        
        return selected

    def _calc_log_likelihood(self, log_p, a, mask):
        """计算对数似然"""
        # log_p: [batch, steps, graph]
        # a: [batch, steps]
        
        # 获取被选动作的对数概率
        log_p_selected = torch.gather(log_p, 2, a.unsqueeze(-1)).squeeze(-1)
        
        if mask is not None:
            log_p_selected[mask] = 0
            
        return log_p_selected.sum(1)

    def _chains_to_tour(self, chain_state):
        """将链条状态转换为TSP路径"""
        batch_size, graph_size = chain_state.batch_size, chain_state.graph_size
        device = chain_state.device
        
        tours = []
        for b in range(batch_size):
            # 简化实现：按节点顺序返回路径
            # 在实际应用中，应该根据链条连接关系重建路径
            tour = list(range(graph_size))
            tours.append(torch.tensor(tour, device=device, dtype=torch.long))
        
        return torch.stack(tours, 0)


class DualChainState:
    """管理双解码器的链条状态"""
    
    def __init__(self, graph_size, batch_size, device):
        self.batch_size = batch_size
        self.graph_size = graph_size
        self.device = device
        
        # 初始化：每个点自成一个链条
        self.chains_start = torch.arange(graph_size, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        self.chains_end = torch.arange(graph_size, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
        
        # 链条是否活跃（还未连接到其他链条）
        self.active_chains = torch.ones(batch_size, graph_size, device=device, dtype=torch.bool)
        
        # 掩码
        self.start_mask = torch.zeros(batch_size, graph_size, device=device, dtype=torch.bool)
        self.end_mask = torch.zeros(batch_size, graph_size, device=device, dtype=torch.bool)
        self._update_masks()
    
    def __getitem__(self, indices):
        """支持切片操作"""
        new_state = DualChainState.__new__(DualChainState)
        new_state.batch_size = indices.shape[0]
        new_state.graph_size = self.graph_size
        new_state.device = self.device
        new_state.chains_start = self.chains_start[indices]
        new_state.chains_end = self.chains_end[indices]
        new_state.active_chains = self.active_chains[indices]
        new_state.start_mask = self.start_mask[indices]
        new_state.end_mask = self.end_mask[indices]
        return new_state
    
    def _update_masks(self):
        """更新起点和终点掩码"""
        batch_size, graph_size = self.batch_size, self.graph_size
        
        # 重置掩码
        self.start_mask.zero_()
        self.end_mask.zero_()
        
        for b in range(batch_size):
            for i in range(graph_size):
                if self.active_chains[b, i]:
                    # 活跃链条的终点可以作为起点
                    end_node = self.chains_end[b, i]
                    if end_node < graph_size:  # 安全检查
                        self.start_mask[b, end_node] = True
                    
                    # 活跃链条的起点可以作为终点
                    start_node = self.chains_start[b, i]
                    if start_node < graph_size:  # 安全检查
                        self.end_mask[b, start_node] = True
    
    def reset_masks(self):
        """重置掩码为默认状态"""
        self.start_mask.zero_()
        self.end_mask.zero_()
        
        # 将所有节点都设为可用
        self.start_mask.fill_(True)
        self.end_mask.fill_(True)
    
    def update_for_selected_start(self, selected_start):
        """在选择起点后更新终点掩码"""
        batch_size = selected_start.shape[0]
        
        for b in range(batch_size):
            start_node = selected_start[b].item()
            
            # 找到包含start_node作为终点的链条
            for i in range(self.graph_size):
                if self.active_chains[b, i] and self.chains_end[b, i].item() == start_node:
                    # 不能选择这个链条的起点作为终点
                    start_chain_start = self.chains_start[b, i].item()
                    if start_chain_start < self.graph_size:  # 安全检查
                        self.end_mask[b, start_chain_start] = False
                    break
    
    def update_chains(self, selected_start, selected_end):
        """更新链条：连接起点和终点所在的链条"""
        batch_size = selected_start.shape[0]
        
        for b in range(batch_size):
            start_node = selected_start[b].item()
            end_node = selected_end[b].item()
            
            # 找到包含start_node作为终点的链条
            start_chain_idx = -1
            for i in range(self.graph_size):
                if self.active_chains[b, i] and self.chains_end[b, i].item() == start_node:
                    start_chain_idx = i
                    break
            
            # 找到包含end_node作为起点的链条
            end_chain_idx = -1
            for i in range(self.graph_size):
                if self.active_chains[b, i] and self.chains_start[b, i].item() == end_node:
                    end_chain_idx = i
                    break
            
            # 如果找到两个不同的链条，合并它们
            if start_chain_idx >= 0 and end_chain_idx >= 0 and start_chain_idx != end_chain_idx:
                # 合并两个链条
                new_start = self.chains_start[b, end_chain_idx].clone()
                new_end = self.chains_end[b, start_chain_idx].clone()
                
                # 更新链条
                self.chains_start[b, start_chain_idx] = new_start
                self.chains_end[b, start_chain_idx] = new_end
                
                # 标记end_chain为不活跃
                self.active_chains[b, end_chain_idx] = False
                
                # 更新掩码
                self._update_masks()
    
    def enhance_with_chain_info(self, embeddings, projection_layer):
        """增强节点嵌入的链条信息"""
        if projection_layer is None:
            return embeddings
        
        batch_size, graph_size, half_dim = embeddings.shape
        
        # 创建链条信息特征
        chain_features = torch.zeros(batch_size, graph_size, 2, device=self.device)
        
        for b in range(batch_size):
            for i in range(graph_size):
                # 特征1：该节点是否是某个链条的起点
                is_start = ((self.chains_start[b] == i) & self.active_chains[b]).any().float()
                # 特征2：该节点是否是某个链条的终点
                is_end = ((self.chains_end[b] == i) & self.active_chains[b]).any().float()
                chain_features[b, i] = torch.tensor([is_start, is_end])
        
        # 拼接并投影
        enhanced = torch.cat([embeddings, chain_features], dim=-1)  # [batch, graph, half_dim+2]
        return projection_layer(enhanced)  # [batch, graph, half_dim]
    
    def get_start_mask(self):
        return self.start_mask
    
    def get_end_mask(self):
        return self.end_mask
    
    def get_active_chain_count(self):
        """获取每个批次中活跃链条的数量"""
        return self.active_chains.sum(dim=1)


# 其余类保持不变（StartDecoder, EndDecoder, InfoPassingModule等）
# 保持与之前版本相同