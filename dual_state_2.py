import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter

class StateDual(NamedTuple):
    # 固定输入
    loc: torch.Tensor  # (batch, n_nodes, 2)
    dist: torch.Tensor # (batch, n_nodes, n_nodes)
    ids: torch.Tensor  # (batch, 1)

    # 动态状态
    # 记录每个节点所属的连通分量ID，初始化为 0..N-1
    group_state: torch.Tensor  # (batch, n_nodes)
    
    # 记录哪些节点还可以做起点(Tail)，哪些可以做终点(Head)
    # 这里的Tail指的是边的出发点(x_k)，Head指的是边的到达点(g_k)
    avail_tails: torch.Tensor # (batch, n_nodes) bool
    avail_heads: torch.Tensor # (batch, n_nodes) bool
    
    i: torch.Tensor  # 当前步数 (batch, 1)

    # 记录生成的边，用于计算最终 Cost
    # prev_tail: torch.Tensor # (batch, n_steps) 记录每一步选的起点
    # prev_head: torch.Tensor # (batch, n_steps) 记录每一步选的终点
    selected_edges_cost: torch.Tensor # (batch, 1) 累加距离

    @staticmethod
    def initialize(loc, visited_dtype=torch.uint8):
        batch_size, n_loc, _ = loc.size()
        device = loc.device

        # 初始化距离矩阵
        dist = (loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1)

        return StateDual(
            loc=loc,
            dist=dist,
            ids=torch.arange(batch_size, dtype=torch.int64, device=device)[:, None],
            # 初始时，每个节点都是一个独立的连通分量，ID即为其索引
            group_state=torch.arange(n_loc, dtype=torch.long, device=device)[None, :].expand(batch_size, n_loc),
            # 初始时，所有点都可以作为边的起点和终点
            avail_tails=torch.ones(batch_size, n_loc, dtype=torch.bool, device=device),
            avail_heads=torch.ones(batch_size, n_loc, dtype=torch.bool, device=device),
            i=torch.zeros(1, dtype=torch.long, device=device),
            selected_edges_cost=torch.zeros(batch_size, 1, device=device)
        )

    def get_tail_mask(self):
        """
        第一步：选择边的起点 (Tail)。
        Mask 掉那些已经做过起点的节点。
        return: (batch, 1, n_nodes) 1 表示被 mask (不可选)
        """
        # avail_tails 为 True 表示可用，Mask 需要 True 表示不可用，取反
        return ~self.avail_tails[:, None, :]

    def get_head_mask(self, selected_tail):
        """
        第二步：选择边的终点 (Head)。
        约束：
        1. 必须是 avail_heads 为 True 的点。
        2. 不能选择与 selected_tail 在同一个连通分量里的点 (防止提前成环)。
           除非这是最后一步 (i == n-1)。
        """
        batch_size, n_nodes = self.avail_tails.size()
        
        # 1. 基本 Mask：不能选已经做过终点的点
        # (batch, n_nodes)
        basic_mask = ~self.avail_heads 

        # 2. 连通性 Mask：不能选 group_id 与 selected_tail 相同的点
        # 获取选中 tail 的 group_id
        # selected_tail: (batch,)
        # self.group_state: (batch, n_nodes)
        tail_groups = self.group_state[self.ids.squeeze(1), selected_tail] # (batch,)
        
        # 找出所有与 tail 同组的节点
        # (batch, n_nodes)
        same_group_mask = (self.group_state == tail_groups[:, None])

        # 如果是最后一步，允许闭环 (实际上必须闭环)，所以最后一步不应用 same_group_mask
        is_last_step = (self.i.item() == (n_nodes - 1))
        
        if is_last_step:
            final_mask = basic_mask
        else:
            final_mask = basic_mask | same_group_mask

        return final_mask[:, None, :] # (batch, 1, n_nodes)

    def update(self, selected_tail, selected_head):
        """
        selected_tail: (batch,) 索引
        selected_head: (batch,) 索引
        """
        batch_size, n_nodes = self.avail_tails.size()

        # 1. 更新 Cost
        # coords: (batch, 2)
        tail_coords = self.loc[self.ids.squeeze(1), selected_tail]
        head_coords = self.loc[self.ids.squeeze(1), selected_head]
        step_dist = (tail_coords - head_coords).norm(p=2, dim=-1)[:, None] # (batch, 1)
        new_cost = self.selected_edges_cost + step_dist

        # 2. 更新可用性 (Mask)
        # 将选中的 tail 设为不可再用
        new_avail_tails = self.avail_tails.scatter(1, selected_tail[:, None], False)
        # 将选中的 head 设为不可再用
        new_avail_heads = self.avail_heads.scatter(1, selected_head[:, None], False)

        # 3. 更新连通分量 (Union-Find Merge)
        # 将 head 所在的组全部合并到 tail 所在的组
        # 获取 tail 的组 ID 和 head 的组 ID
        tail_gid = self.group_state[self.ids.squeeze(1), selected_tail] # (batch,)
        head_gid = self.group_state[self.ids.squeeze(1), selected_head] # (batch,)
        
        # 创建 mask：找出所有属于 head 组的节点
        # (batch, n_nodes)
        mask_nodes_in_head_group = (self.group_state == head_gid[:, None])
        
        # 使用 where 更新：如果节点属于 head 组，则将其 ID 改为 tail 组 ID，否则保持不变
        # 这样完成了 Union 操作
        new_group_state = torch.where(
            mask_nodes_in_head_group,
            tail_gid[:, None], # True 的位置填 tail_gid
            self.group_state   # False 的位置保持原样
        )

        return self._replace(
            avail_tails=new_avail_tails,
            avail_heads=new_avail_heads,
            group_state=new_group_state,
            selected_edges_cost=new_cost,
            i=self.i + 1
        )

    def all_finished(self):
        return self.i.item() >= self.loc.size(-2)

    def get_final_cost(self):
        return self.selected_edges_cost