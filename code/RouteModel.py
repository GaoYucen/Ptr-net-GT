# RL_ProbNetwork.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class RouteGenerator(nn.Module):
    def __init__(self,
                 node_dim=2,
                 embed_dim=128,
                 hidden_dim=256,
                 n_layers=1,
                 dropout=0.):
        super(RouteGenerator, self).__init__()

        # 节点坐标嵌入层
        self.node_embed = nn.Linear(node_dim, embed_dim)

        # 状态跟踪LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim + 1,  # 嵌入坐标 + 时间步特征
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # 概率输出层
        self.prob_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 每个节点的评分
        )

        # 时间步嵌入
        self.step_embed = nn.Embedding(100, 1)  # 支持最多100个节点

        # 初始化隐藏状态
        self.h_init = nn.Parameter(torch.randn(n_layers, 1, hidden_dim))
        self.c_init = nn.Parameter(torch.randn(n_layers, 1, hidden_dim))

    def forward(self, coordinates, mask, hidden=None):
        """
        coordinates: (batch_size, seq_len, 2)
        mask: (batch_size, seq_len) - 已访问节点为1
        """
        batch_size, seq_len, _ = coordinates.shape

        # 节点嵌入
        node_embeds = self.node_embed(coordinates)  # (B, L, E)

        # 生成时间步特征
        step_ids = torch.arange(seq_len, device=coordinates.device).unsqueeze(0)
        step_feats = self.step_embed(step_ids.expand(batch_size, -1))  # (B, L, 1)

        # 拼接特征
        combined = torch.cat([node_embeds, step_feats], dim=-1)  # (B, L, E+1)

        # LSTM处理序列
        if hidden is None:
            # 扩展初始隐藏状态
            h = self.h_init.repeat(1, batch_size, 1)
            c = self.c_init.repeat(1, batch_size, 1)
            hidden = (h, c)

        lstm_out, hidden = self.lstm(combined, hidden)  # (B, L, H)

        # 计算节点评分
        node_scores = self.prob_out(lstm_out).squeeze(-1)  # (B, L)

        # 应用mask
        node_scores = node_scores.masked_fill(mask.bool(), float('-inf'))

        # 转换为概率分布
        prob_matrix = F.softmax(node_scores, dim=-1)

        return prob_matrix, hidden

    def init_hidden(self, batch_size):
        h = self.h_init.repeat(1, batch_size, 1)
        c = self.c_init.repeat(1, batch_size, 1)
        return (h, c)