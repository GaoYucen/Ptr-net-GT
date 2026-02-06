import os
import json
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from utils import load_problem

# ================= 配置区域 =================


OLD_MODEL_PATH = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356\epoch-26.pt'
OLD_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356\args.json'  # 必须指定对应的 args.json


NEW_MODEL_PATH = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\best-model_91.pt'
NEW_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\args.json'  # 必须指定对应的 args.json

GRAPH_SIZE = 50

# 【核心修改】设置 Cluster 类型数据的占比 (0.0 ~ 1.0)
# 0.0 = 全是均匀分布 (Uniform) -> 预期旧方法 Cost 更低
# 0.5 = 一半一半 (Mixed) -> 预期新方法 Cost 可能反超
# 1.0 = 全是聚类分布 (Cluster) -> 预期新方法 Cost 显著更低
CLUSTER_PROPORTION = 0.0

NOISE_LEVELS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2,0.3,0.4,0.5,0.6]
NUM_BASE_INSTANCES = 200  # 总样本数
NUM_NOISE_TRIALS = 50     # 扰动次数
BATCH_SIZE = 100
SEED = 1234
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ===========================================

from nets.attention_model import AttentionModel as ModelOld
try:
    from nets.attention_model_dual_crossStep import AttentionModel as ModelNew
except ImportError:
    from nets.attention_model import AttentionModel as ModelNew

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

def evaluate_noise_impact(model, base_data, noise_level):
    """
    对 base_data 进行扰动测试
    """
    batch_size = base_data.size(0)
    # 扩展数据: (N, Trials, Size, 2)
    expanded = base_data.unsqueeze(1).repeat(1, NUM_NOISE_TRIALS, 1, 1)
    
    # 生成噪声
    noise = torch.randn_like(expanded) * noise_level
    noisy_data = torch.clamp(expanded + noise, 0.0, 1.0)
    
    # 展平以便 batch 处理
    flattened_data = noisy_data.view(-1, GRAPH_SIZE, 2)
    
    dataset = TensorDataset(flattened_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    all_costs = []
    with torch.no_grad():
        for batch, in loader:
            costs, _ = model(batch.to(DEVICE))
            all_costs.append(costs.cpu())
            
    all_costs = torch.cat(all_costs) 
    
    # 重塑回 (N, Trials)
    reshaped_costs = all_costs.view(batch_size, NUM_NOISE_TRIALS)
    
    # 1. 性能指标 (Mean)
    mean_performance = reshaped_costs.mean().item()
    
    # 2. 稳定性指标 (Std over trials)
    stability = reshaped_costs.std(dim=1).mean().item()
    
    return mean_performance, stability

def main():
    torch.manual_seed(SEED)
    
    print("Loading models...")
    try:
        model_old = load_model(OLD_MODEL_PATH, OLD_ARGS_PATH, ModelOld)
        model_new = load_model(NEW_MODEL_PATH, NEW_ARGS_PATH, ModelNew)
    except Exception as e:
        print(f"Error: {e}")
        return

    # --- 核心修改：根据比例生成数据 ---
    num_cluster = int(NUM_BASE_INSTANCES * CLUSTER_PROPORTION)
    num_uniform = NUM_BASE_INSTANCES - num_cluster
    
    print(f"\nGenerating Data with CLUSTER_PROPORTION = {CLUSTER_PROPORTION}")
    print(f"  -> Uniform Samples: {num_uniform}")
    print(f"  -> Cluster Samples: {num_cluster}")
    
    data_parts = []
    
    # 1. 生成 Uniform 数据
    if num_uniform > 0:
        data_u = torch.rand(num_uniform, GRAPH_SIZE, 2)
        data_parts.append(data_u)
        
    # 2. 生成 Cluster 数据
    if num_cluster > 0:
        n_centers = 3
        centers = torch.rand(num_cluster, n_centers, 2)
        batch_idx = torch.arange(num_cluster).view(-1, 1).expand(-1, GRAPH_SIZE)
        cluster_idx = torch.randint(0, n_centers, (num_cluster, GRAPH_SIZE))
        
        # 选取中心并加噪生成点
        data_c = centers[batch_idx, cluster_idx] + torch.randn(num_cluster, GRAPH_SIZE, 2) * 0.07
        data_c = torch.clamp(data_c, 0.0, 1.0)
        data_parts.append(data_c)
    
    # 合并
    if len(data_parts) > 0:
        base_data = torch.cat(data_parts, dim=0)
    else:
        print("Error: No data generated.")
        return

    # --- 开始测试 ---
    results = {
        'noise': NOISE_LEVELS,
        'old_mean': [], 'new_mean': [],
        'old_std': [], 'new_std': []
    }

    print(f"\n{'Noise':<8} | {'Old Mean':<10} | {'New Mean':<10} | {'Old Std':<10} | {'New Std':<10}")
    print("-" * 60)

    for noise in NOISE_LEVELS:
        mean_old, std_old = evaluate_noise_impact(model_old, base_data, noise)
        mean_new, std_new = evaluate_noise_impact(model_new, base_data, noise)
        
        results['old_mean'].append(mean_old)
        results['new_mean'].append(mean_new)
        results['old_std'].append(std_old)
        results['new_std'].append(std_new)
        
        print(f"{noise:<8.3f} | {mean_old:<10.4f} | {mean_new:<10.4f} | {std_old:<10.4f} | {std_new:<10.4f}")

    # --- 绘图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 标题加入比例信息
    fig.suptitle(f'Noise Robustness Test (Cluster Ratio: {CLUSTER_PROPORTION*100:.0f}%)', fontsize=16)
    
    # Cost
    ax1.plot(results['noise'], results['old_mean'], 'o-', label='Old Model', linewidth=2)
    ax1.plot(results['noise'], results['new_mean'], 's-', label='New Model', linewidth=2)
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('Avg Cost')
    ax1.set_title('Performance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Stability
    ax2.plot(results['noise'], results['old_std'], 'o--', label='Old Model', linewidth=2)
    ax2.plot(results['noise'], results['new_std'], 's--', label='New Model', linewidth=2)
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('Std Dev')
    ax2.set_title('Stability')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    filename = f'noise_trend_ratio_{CLUSTER_PROPORTION}.png'
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"\n>> Plot saved to '{filename}'")

if __name__ == "__main__":
    main()