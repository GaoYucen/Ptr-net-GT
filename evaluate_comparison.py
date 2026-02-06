import os
import json
import torch
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from utils import load_problem

# ================= 配置区域 =================
# 1. 旧方法 (Baseline/Rollout) 的路径
OLD_MODEL_PATH = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356\epoch-26.pt'
OLD_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356\args.json'  # 必须指定对应的 args.json

# 2. 新方法 (Dual/CrossStep) 的路径
NEW_MODEL_PATH = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\best-model_91.pt'
NEW_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\args.json'  # 必须指定对应的 args.json

# 3. 测试设置
GRAPH_SIZE = 50     # 图的大小
NUM_TEST_SAMPLES = 100   # 【新增】用于统计评估的大规模测试样本数
BATCH_SIZE = 256          # 批量大小，防显存溢出
NUM_VISUALIZATION = 3     # 可视化的样本数量 (从测试集中取前3个)
SEED = 603               # 随机种子
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ===========================================

# 动态导入模型类
from nets.attention_model import AttentionModel as ModelOld
try:
    from nets.attention_model_dual_crossStep import AttentionModel as ModelNew
except ImportError:
    print("Warning: Could not import 'nets.attention_model_crossStep'. Using 'nets.attention_model' for new model.")
    from nets.attention_model import AttentionModel as ModelNew

def load_args(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Args file not found: {path}")
    with open(path, 'r') as f:
        args = json.load(f)
    return args

def fix_state_dict_keys(state_dict):
    """修复旧模型权重中 SkipConnection 缺失 .module 的问题"""
    new_state_dict = {}
    for key, value in state_dict.items():
        parts = key.split('.')
        if parts[0] == 'module':
            parts = parts[1:]
        
        if len(parts) > 3 and parts[0] == 'embedder' and parts[1] == 'layers':
            sub_layer_idx = parts[3]
            if sub_layer_idx in ['0', '2'] and parts[4] != 'module':
                new_key_parts = parts[:4] + ['module'] + parts[4:]
                new_key = ".".join(new_key_parts)
                new_state_dict[new_key] = value
                continue

        key_without_dp = ".".join(parts)
        new_state_dict[key_without_dp] = value
    return new_state_dict

def load_model(model_path, args_path, model_class):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading configuration from {args_path}...")
    args = load_args(args_path)
    
    problem = load_problem(args.get('problem', 'tsp'))

    model = model_class(
        embedding_dim=args.get('embedding_dim', 128),
        hidden_dim=args.get('hidden_dim', 128),
        problem=problem,
        n_encode_layers=args.get('n_encode_layers', 3),
        mask_inner=True,
        mask_logits=True,
        normalization=args.get('normalization', 'batch'),
        tanh_clipping=args.get('tanh_clipping', 10.),
        checkpoint_encoder=False,
        shrink_size=None
    ).to(DEVICE)

    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    raw_state_dict = checkpoint['model']
    final_state_dict = fix_state_dict_keys(raw_state_dict)

    try:
        model.load_state_dict(final_state_dict, strict=True)
    except RuntimeError as e:
        print(f"\n[Warning] Strict loading failed. Retrying with strict=False.")
        model.load_state_dict(final_state_dict, strict=False)

    model.eval()
    model.set_decode_type("greedy")
    return model

def evaluate_dataset(model, dataset, batch_size, device):
    """
    对整个数据集进行评估，返回所有成本和第一批次的路径（用于可视化）
    """
    dataloader = DataLoader(TensorDataset(dataset), batch_size=batch_size)
    all_costs = []
    first_batch_pi = None

    with torch.no_grad():
        for i, (batch_data,) in enumerate(dataloader):
            batch_data = batch_data.to(device)
            cost, _, pi = model(batch_data, return_pi=True)
            all_costs.append(cost)
            
            if i == 0:
                first_batch_pi = pi.cpu()

    return torch.cat(all_costs), first_batch_pi

def plot_tour(ax, coords, tour, cost, title_prefix, is_dual=False):
    x = coords[:, 0]
    y = coords[:, 1]
    
    ax.scatter(x, y, c='blue', s=50, zorder=2)
    ax.scatter(x[0], y[0], c='red', s=80, marker='*', zorder=3, label='Start')

    if is_dual:
        # 新模型 (Edge List)
        tails = tour[0::2]
        heads = tour[1::2]
        for t, h in zip(tails, heads):
            p1 = coords[t]
            p2 = coords[h]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='black', alpha=0.8, zorder=1)
    else:
        # 旧模型 (Sequence)
        tour_indices = list(tour)
        if len(tour_indices) > 0 and tour_indices[-1] != tour_indices[0]:
            tour_indices.append(tour_indices[0])
            
        for i in range(len(tour_indices) - 1):
            p1 = coords[tour_indices[i]]
            p2 = coords[tour_indices[i+1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='gray', alpha=0.7, zorder=1)
            # 箭头
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            dx = (p2[0] - p1[0]) * 0.05
            dy = (p2[1] - p1[1]) * 0.05
            ax.arrow(mid_x, mid_y, dx, dy, shape='full', lw=0, 
                     length_includes_head=True, head_width=0.02, color='gray')

    ax.set_title(f"{title_prefix}\nCost: {cost:.4f}")
    ax.axis('off')

def main():
    torch.manual_seed(SEED)
    
    print(f"\nGenerating {NUM_TEST_SAMPLES} random TSP instances (Size {GRAPH_SIZE})...")
    # 生成大规模测试集
    data = torch.rand(NUM_TEST_SAMPLES, GRAPH_SIZE, 2) # 先放在CPU上
    
    try:
        print("\n--- Loading OLD Model ---")
        model_old = load_model(OLD_MODEL_PATH, OLD_ARGS_PATH, ModelOld)
        
        print("\n--- Loading NEW Model ---")
        model_new = load_model(NEW_MODEL_PATH, NEW_ARGS_PATH, ModelNew)
    except Exception as e:
        print(f"\n[Error] {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 批量推理与统计
    print(f"\nEvaluating models on {NUM_TEST_SAMPLES} samples (Batch size: {BATCH_SIZE})...")
    
    # 评估旧模型
    costs_old, pi_old_batch = evaluate_dataset(model_old, data, BATCH_SIZE, DEVICE)
    # 评估新模型
    costs_new, pi_new_batch = evaluate_dataset(model_new, data, BATCH_SIZE, DEVICE)

    # 4. 计算统计数据
    mean_old = costs_old.mean().item()
    var_old = costs_old.var().item()
    std_old = costs_old.std().item()

    mean_new = costs_new.mean().item()
    var_new = costs_new.var().item()
    std_new = costs_new.std().item()

    print("\n" + "="*40)
    print(f" STATISTICAL RESULTS (N={NUM_TEST_SAMPLES})")
    print("="*40)
    print(f"{'Metric':<15} | {'Old Model':<15} | {'New Model':<15}")
    print("-" * 50)
    print(f"{'Mean Cost':<15} | {mean_old:<15.5f} | {mean_new:<15.5f}")
    print(f"{'Variance':<15} | {var_old:<15.5f} | {var_new:<15.5f}")
    print(f"{'Std Dev':<15} | {std_old:<15.5f} | {std_new:<15.5f}")
    print("="*40)

    # 5. 可视化前 NUM_VISUALIZATION 个样本
    print(f"\nPlotting top {NUM_VISUALIZATION} results...")
    
    # 取出前N个用于画图的数据 (已经在 evaluate_dataset 中取回了第一批次的 pi)
    # data 的前 N 个
    vis_data = data[:NUM_VISUALIZATION]
    vis_pi_old = pi_old_batch[:NUM_VISUALIZATION]
    vis_pi_new = pi_new_batch[:NUM_VISUALIZATION]
    vis_cost_old = costs_old[:NUM_VISUALIZATION]
    vis_cost_new = costs_new[:NUM_VISUALIZATION]

    fig, axes = plt.subplots(NUM_VISUALIZATION, 2, figsize=(12, 5 * NUM_VISUALIZATION))
    if NUM_VISUALIZATION == 1:
        axes = axes.reshape(1, -1)

    for i in range(NUM_VISUALIZATION):
        coords = vis_data[i].numpy()
        
        # 左侧：旧方法
        plot_tour(
            axes[i, 0], 
            coords, 
            vis_pi_old[i].numpy(), 
            vis_cost_old[i].item(), 
            "Old Method (Sequence)", 
            is_dual=False
        )
        
        # 右侧：新方法
        plot_tour(
            axes[i, 1], 
            coords, 
            vis_pi_new[i].numpy(), 
            vis_cost_new[i].item(), 
            "New Method (Dual Edges)", 
            is_dual=True
        )

    plt.tight_layout()
    save_filename = 'tsp_stats_comparison.png'
    plt.savefig(save_filename, dpi=300)
    print(f"\n[Success] Comparison plot saved to '{save_filename}'.")
    plt.show()

if __name__ == "__main__":
    main()