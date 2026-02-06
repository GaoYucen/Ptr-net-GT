import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple
from utils.tensor_functions import compute_in_batches

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel
from utils.beam_search import CachedLookup
from utils.functions import sample_many

# 引入之前的 StateDual (确保你的目录下有 dual_state_2.py)
from dual_state_2 import StateDual

def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)

class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)
        return AttentionModelFixed(
            node_embeddings=self.node_embeddings[key],
            context_node_projected=self.context_node_projected[key],
            glimpse_key=self.glimpse_key[:, key], 
            glimpse_val=self.glimpse_val[:, key], 
            logit_key=self.logit_key[key]
        )

class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.tanh_clipping = tanh_clipping
        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        node_dim = 2  # x, y

        # Embedding for Input Nodes
        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # Projections for Attention (Keys/Values/Logits)
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # --- Dual Decoder Projections (Modified) ---
        
        # 1. Tail Selection Decoder (Modified: NO Cross-Step Context)
        # Input: Fixed Context (Global) ONLY -> Output: Query for Tail
        # 【修改点 1】: 输入维度改为 embedding_dim (原来是 2 * embedding_dim)
        self.project_step_context_tail = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_out_tail = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # 【修改点 2】: 删除了 self.W_placeholder_tail，因为不再需要上一时刻的 Embedding
        
        # 2. Head Selection Decoder
        # Input: Fixed Context (Global) + Selected Tail Embedding -> Output: Query for Head
        # 这里保持不变，因为选终点必须依赖于当前选的起点
        self.project_step_context_head = nn.Linear(2 * embedding_dim, embedding_dim, bias=False)
        self.project_out_head = nn.Linear(embedding_dim, embedding_dim, bias=False)
        

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:
            self.temp = temp

    def forward(self, input, return_pi=False):
        """
        :param input: (batch_size, graph_size, node_dim)
        """
        if self.checkpoint_encoder and self.training:
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))

        _log_p, pi = self._inner(input, embeddings)

        cost = self._calc_cost(input, pi)
        ll = self._calc_log_likelihood(_log_p, pi, None)
        
        if return_pi:
            return cost, ll, pi

        return cost, ll

    def _init_embed(self, input):
        return self.init_embed(input)

    def _inner(self, input, embeddings):
        
        state = StateDual.initialize(input)
        
        outputs = [] 
        sequences = [] 

        fixed = self._precompute(embeddings)
        batch_size = state.ids.size(0)

        # 【修改点 3】: 移除了 last_head_embedding 的初始化

        i = 0
        while not state.all_finished():
            
            # --- Step 1: Select Tail (Independent of previous steps) ---
            
            # Context: Global Graph Context ONLY
            # fixed.context_node_projected: (batch, 1, embed)
            # 【修改点 4】: 仅仅使用 fixed context，不再拼接 last_head_embedding
            tail_context_input = fixed.context_node_projected 
            
            # Project to obtain dynamic query for Tail
            query_tail = self.project_step_context_tail(tail_context_input) # (batch, 1, embed)
            
            mask_tail = state.get_tail_mask()
            
            log_p_tail, _ = self._get_log_p(fixed, query_tail, mask_tail, self.project_out_tail)
            
            selected_tail = self._select_node(log_p_tail.exp()[:, 0, :], mask_tail[:, 0, :])
            
            # --- Step 2: Select Head (Dependent on current Tail) ---
            
            # Context: Global Graph Context + Current Selected Tail
            tail_embedding = torch.gather(
                embeddings, 
                1, 
                selected_tail[:, None, None].expand(batch_size, 1, self.embedding_dim)
            )
            
            head_context_input = torch.cat((fixed.context_node_projected, tail_embedding), dim=-1)
            
            query_head = self.project_step_context_head(head_context_input)
            
            mask_head = state.get_head_mask(selected_tail)
            
            log_p_head, _ = self._get_log_p(fixed, query_head, mask_head, self.project_out_head)
            
            selected_head = self._select_node(log_p_head.exp()[:, 0, :], mask_head[:, 0, :])
            
            # --- Update State ---
            
            state = state.update(selected_tail, selected_head)

            # 【修改点 5】: 移除了 last_head_embedding 的更新逻辑

            # --- Store Outputs ---
            outputs.append(log_p_tail[:, 0, :])
            outputs.append(log_p_head[:, 0, :])
            sequences.append(selected_tail)
            sequences.append(selected_head)

            i += 1

        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def _get_log_p(self, fixed, query, mask, project_out_layer):
        glimpse_K, glimpse_V, logit_K = fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask, project_out_layer)
        log_p = torch.log_softmax(log_p / self.temp, dim=-1)
        return log_p, mask

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask, project_out_layer):
        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        
        if self.mask_inner:
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        glimpse = project_out_layer(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size)
        )

        final_Q = glimpse
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _select_node(self, probs, mask):
        assert (probs == probs).all(), "Probs should not contain any nans"
        if self.decode_type == "greedy":
            _, selected = probs.max(1)
        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)
        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings):
        graph_embed = embeddings.mean(1)
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]
        
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(embeddings[:, None, :, :]).chunk(3, dim=-1)

        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps=1),
            self._make_heads(glimpse_val_fixed, num_steps=1),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _make_heads(self, v, num_steps=None):
        return (
            v.contiguous().view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4) 
        )

    def _calc_log_likelihood(self, _log_p, a, mask):
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)
        return log_p.sum(1)

    def _calc_cost(self, input, pi):
        tails = pi[:, 0::2]
        heads = pi[:, 1::2]
        
        loc = input
        batch_size, n_nodes, _ = loc.size()
        
        tail_coords = torch.gather(loc, 1, tails[:, :, None].expand(batch_size, n_nodes, 2))
        head_coords = torch.gather(loc, 1, heads[:, :, None].expand(batch_size, n_nodes, 2))
        
        dists = (tail_coords - head_coords).norm(p=2, dim=-1)
        return dists.sum(1)