# valid_R.py
# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys

sys.path.append('code')
from RouteModel import RouteGenerator  # 修改模型导入
from Data_Generator import TSPDataset
from config import get_config


def get_length(point, solution):
    """计算路径长度（优化版）"""
    seq = np.roll(solution, -1)
    diff = point[seq] - point[solution]
    return np.linalg.norm(diff, axis=1).sum()


# %% 主验证流程
if __name__ == '__main__':
    params, _ = get_config()

    # %% 设备设置（与训练代码一致）
    device = torch.device('cuda' if torch.cuda.is_available() and params.gpu else 'mps' if params.gpu else 'cpu')
    print(f'Using device: {device}')

    # %% 初始化模型
    model = RouteGenerator(
        node_dim=2,
        embed_dim=params.embedding_size,
        hidden_dim=params.hiddens,
        n_layers=params.nof_lstms,
        dropout=params.dropout
    ).to(device)

    # %% 加载训练好的模型
    model_path = f'param/reinforce_param_{params.nof_points}_{params.nof_epoch}.pkl'  # 匹配新保存路径
    # model_path = f'param/best_loss_reinforce_param_{params.nof_points}_{params.nof_epoch}.pkl'  # 匹配新保存路径
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f'Successfully loaded model from {model_path}')

    # %% 数据加载
    test_dataset = np.load('data/test.npy', allow_pickle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=params.batch_size,
                             shuffle=False)

    # %% 验证指标
    total_error = 0.0
    sample_count = 0
    progress = tqdm(test_loader, desc='Validating')

    with torch.no_grad():
        for batch in progress:
            points = batch['Points'].float().to(device)  # [B, L, 2]
            solutions_opt = batch['Solutions'].numpy()  # 真实最优解

            batch_size, seq_len, _ = points.shape

            # 逐步生成路径
            mask = torch.zeros(batch_size, seq_len, device=device)
            solutions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            hidden = model.init_hidden(batch_size)

            # 起始节点设为0
            solutions[:, 0] = 0
            mask[torch.arange(batch_size), 0] = 1

            # 生成路径
            for step in range(seq_len-1):
                prob_matrix, hidden = model(points, mask, hidden)

                # 贪心选择
                actions = prob_matrix.argmax(dim=-1)

                # 记录路径
                solutions[:, step+1] = actions

                # 更新mask（最后一步保持开放）
                if step < seq_len - 1:
                    mask = mask.scatter(1, actions.unsqueeze(1), 1)

            # 转换为numpy计算
            points_np = points.cpu().numpy()
            solutions_np = solutions.cpu().numpy()

            # 计算批次误差
            batch_error = 0.0
            for i in range(batch_size):
                pred_length = get_length(points_np[i], solutions_np[i])
                opt_length = get_length(points_np[i], solutions_opt[i])

                error = (pred_length - opt_length) / opt_length * 100
                batch_error += error

                # 可选：打印详细信息
                # print(f'Sample {sample_count+i}: Pred={pred_length:.2f}, Opt={opt_length:.2f}, Error={error:.2f}%')

            total_error += batch_error
            sample_count += batch_size
            progress.set_postfix(avg_error=f'{total_error / sample_count:.2f}%')

    # %% 最终结果
    final_error = total_error / sample_count
    print(f'\nValidation Complete. Average Error: {final_error:.2f}%')
    print(f'Tested on {sample_count} samples')