import os
import json
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from utils import load_problem


# ================= 配置区域 =================

OLD_MODEL_PATH = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356\epoch-49.pt'
OLD_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356\args.json'  # 必须指定对应的 args.json


NEW_MODEL_PATH = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_20\tsp20_dual_warmup_20260123T160816\best-model.pt'
NEW_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_20\tsp20_dual_warmup_20260123T160816\args.json'  # 必须指定对应的 args.json

# 2. 测试配置
GRAPH_SIZE = 50           # 图的大小
NUM_TEST_SAMPLES = 3000   # 统计评估用的样本总数
NUM_VIS_SAMPLES = 3       # 最多画几张图 (只画新模型表现更好的)
BATCH_SIZE = 256
SEED = 333
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 分布类型列表
DISTRIBUTIONS = ['uniform', 'clustered', 'gaussian', 'mixed']
# DISTRIBUTIONS = ['clustered']
# ===========================================

# 动态导入模型
from nets.attention_model import AttentionModel as ModelOld
try:
    from nets.attention_model_dual_no_context import AttentionModel as ModelNew
except ImportError:
    print("Warning: Using 'nets.attention_model' for New Model (File not found).")
    from nets.attention_model import AttentionModel as ModelNew

# --- 数据生成器 ---
def generate_data(dist_type, num_samples, graph_size):
    """根据分布类型生成数据"""
    if dist_type == 'uniform':
        return torch.rand(num_samples, graph_size, 2)
    
    elif dist_type == 'clustered':
        n_clusters = 3
        centers = torch.rand(num_samples, n_clusters, 2)
        batch_indices = torch.arange(num_samples).view(-1, 1).expand(-1, graph_size)
        cluster_indices = torch.randint(0, n_clusters, (num_samples, graph_size))
        selected_centers = centers[batch_indices, cluster_indices]
        noise = torch.randn(num_samples, graph_size, 2) * 0.07
        return torch.clamp(selected_centers + noise, 0.0, 1.0)
    
    elif dist_type == 'gaussian':
        return torch.clamp(torch.randn(num_samples, graph_size, 2) * 0.15 + 0.5, 0.0, 1.0)

    elif dist_type == 'mixed':
        split = graph_size // 2
        part1 = torch.rand(num_samples, split, 2)
        center = torch.rand(num_samples, 1, 2)
        noise = torch.randn(num_samples, graph_size - split, 2) * 0.05
        part2 = torch.clamp(center + noise, 0.0, 1.0)
        return torch.cat((part1, part2), dim=1)
    else:
        raise ValueError(f"Unknown distribution: {dist_type}")

# --- 模型加载工具 ---
def load_args(path):
    with open(path, 'r') as f: return json.load(f)

def fix_state_dict_keys(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        parts = key.split('.')
        if parts[0] == 'module': parts = parts[1:]
        if len(parts) > 3 and parts[0] == 'embedder' and parts[1] == 'layers':
            if parts[3] in ['0', '2'] and parts[4] != 'module':
                parts.insert(4, 'module')
                new_state_dict[".".join(parts)] = value
                continue
        new_state_dict[".".join(parts)] = value
    return new_state_dict

def load_model(model_path, args_path, model_class):
    if not os.path.exists(model_path): raise FileNotFoundError(model_path)
    args = load_args(args_path)
    problem = load_problem(args.get('problem', 'tsp'))
    
    model = model_class(
        embedding_dim=args.get('embedding_dim', 128),
        hidden_dim=args.get('hidden_dim', 128),
        problem=problem,
        n_encode_layers=args.get('n_encode_layers', 3),
        mask_inner=True, mask_logits=True, normalization=args.get('normalization', 'batch'),
        tanh_clipping=10., checkpoint_encoder=False, shrink_size=None
    ).to(DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(fix_state_dict_keys(checkpoint['model']), strict=False)
    model.eval()
    model.set_decode_type("greedy")
    return model

# --- 评估核心 ---
def evaluate(model, data):
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    costs = []
    # 我们需要所有样本的路径用于筛选绘图
    all_pi = [] 
    
    with torch.no_grad():
        for i, (batch,) in enumerate(loader):
            batch = batch.to(DEVICE)
            cost, _, pi = model(batch, return_pi=True)
            costs.append(cost)
            all_pi.append(pi.cpu())
            
    return torch.cat(costs), torch.cat(all_pi)

# --- 可视化单个样本 ---
def plot_sample_comparison(dist_name, sample_idx, real_idx, coords, pi_old, cost_old, pi_new, cost_new):
    """
    为单个样本生成一张对比图 (左旧右新)
    sample_idx: 当前是第几张图 (0, 1, 2)
    real_idx: 在原始数据集中的索引
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 计算 GAP
    gap = (cost_new - cost_old) / cost_old * 100
    gap_str = f"{gap:.2f}%" 

    # --- 左图：旧模型 ---
    ax = axes[0]
    ax.scatter(coords[:,0], coords[:,1], c='blue', zorder=2)
    ax.scatter(coords[0,0], coords[0,1], c='red', marker='*', s=150, zorder=3, label='Start')
    
    tour = list(pi_old)
    if tour[-1] != tour[0]: tour.append(tour[0])
    for j in range(len(tour)-1):
        p1, p2 = coords[tour[j]], coords[tour[j+1]]
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], c='gray', alpha=0.7, zorder=1)
        # 画箭头
        mid_x, mid_y = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
        dx, dy = (p2[0]-p1[0])*0.05, (p2[1]-p1[1])*0.05
        ax.arrow(mid_x, mid_y, dx, dy, shape='full', lw=0, length_includes_head=True, head_width=0.02, color='gray')
        
    ax.set_title(f"Old Method (Sequence)\nCost: {cost_old:.4f}", fontsize=14)
    ax.axis('off')

    # --- 右图：新模型 ---
    ax = axes[1]
    ax.scatter(coords[:,0], coords[:,1], c='blue', zorder=2)
    ax.scatter(coords[0,0], coords[0,1], c='red', marker='*', s=150, zorder=3)
    
    # 新模型输出 Edge List
    tails, heads = pi_new[0::2], pi_new[1::2]
    for t, h in zip(tails, heads):
        p1, p2 = coords[t], coords[h]
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], c='black', alpha=0.9, linewidth=1.5, zorder=1)
        
    ax.set_title(f"New Method (Dual Edges)\nCost: {cost_new:.4f} (Gap: {gap_str})", fontsize=14, color='green')
    ax.axis('off')

    # 保存图片
    save_dir = 'vis_results_better'
    os.makedirs(save_dir, exist_ok=True)
    # 文件名增加 ID 以便溯源
    filename = f"{save_dir}/vis_{dist_name}_better_top{sample_idx+1}_id{real_idx}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    torch.manual_seed(SEED)
    
    # 1. 加载模型
    try:
        print("Loading models...")
        model_old = load_model(OLD_MODEL_PATH, OLD_ARGS_PATH, ModelOld)
        model_new = load_model(NEW_MODEL_PATH, NEW_ARGS_PATH, ModelNew)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 打印表头
    header = f"{'Distribution':<12} | {'Old Mean':<10} | {'Old Var':<10} | {'New Mean':<10} | {'New Var':<10} | {'Gap (%)':<8}"
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'-'*len(header)}")

    # 2. 循环测试分布
    for dist_name in DISTRIBUTIONS:
        # 生成测试数据
        data = generate_data(dist_name, NUM_TEST_SAMPLES, GRAPH_SIZE)
        
        # 评估 (获取所有 Cost 和路径)
        costs_old, all_pi_old = evaluate(model_old, data)
        costs_new, all_pi_new = evaluate(model_new, data)
        
        # --- 统计分析 (Mean & Variance) ---
        mean_old = costs_old.mean().item()
        var_old  = costs_old.var().item()
        
        mean_new = costs_new.mean().item()
        var_new  = costs_new.var().item()
        
        gap = ((mean_new - mean_old) / mean_old) * 100
        
        print(f"{dist_name:<12} | {mean_old:<10.4f} | {var_old:<10.4f} | {mean_new:<10.4f} | {var_new:<10.4f} | {gap:<8.2f}")
        
        # --- 智能绘图筛选 ---
        # 找出新方法表现更好 (New Cost < Old Cost) 的索引
        better_mask = costs_new < costs_old
        better_indices = torch.nonzero(better_mask).flatten()
        
        num_better = len(better_indices)
        
        if num_better > 0:
            # 按 GAP 大小排序 (差距越大越好，Gap 越负越好)
            # 计算这些样本的 gap
            better_gaps = (costs_new[better_indices] - costs_old[better_indices])
            # 排序：gap 越小(越负)说明提升越大，所以按升序排
            _, sorted_sub_indices = torch.sort(better_gaps, descending=False)
            
            # 取前 NUM_VIS_SAMPLES 个 (不足则全取)
            top_k = min(NUM_VIS_SAMPLES, num_better)
            selected_indices = better_indices[sorted_sub_indices[:top_k]]
            
            print(f"  -> Found {num_better} better samples. Plotting top {top_k} best cases...")
            
            for rank, idx in enumerate(selected_indices):
                plot_sample_comparison(
                    dist_name, 
                    rank,             # 排名 (0, 1, 2)
                    idx.item(),       # 原始 ID
                    data[idx].numpy(), 
                    all_pi_old[idx].numpy(), 
                    costs_old[idx].item(), 
                    all_pi_new[idx].numpy(), 
                    costs_new[idx].item()
                )
        else:
            print("  -> No samples found where New Model is better than Old Model. Skipping plots.")

    print(f"{'='*len(header)}")
    print(f"Visualization images saved to 'vis_results_better' folder.")

if __name__ == "__main__":
    main()