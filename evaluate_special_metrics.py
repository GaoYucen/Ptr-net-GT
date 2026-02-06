import os
import json
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from utils import load_problem



OLD_MODEL_PATH = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356\epoch-26.pt'
OLD_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356\args.json'  # 必须指定对应的 args.json


NEW_MODEL_PATH = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\best-model_91.pt'
NEW_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\args.json'  # 必须指定对应的 args.json

GRAPH_SIZE = 50
NUM_SAMPLES = 100       # 测试样本数
NOISE_LEVEL = 0.05    # 坐标扰动幅度
NUM_NOISE_TRIALS = 50   # 对每个样本扰动多少次
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

def get_max_edge_length(data, pi):
    """计算路径中最长的一条边 (Bottleneck Metric)"""
    # data: (B, N, 2)
    # pi: (B, N)
    batch_size, n, _ = data.shape
    max_edges = []
    
    data_np = data.cpu().numpy()
    pi_np = pi.cpu().numpy()
    
    for i in range(batch_size):
        coords = data_np[i]
        tour = list(pi_np[i])
        if tour[-1] != tour[0]: tour.append(tour[0])
        
        # 计算所有边的长度
        edge_lengths = []
        for j in range(len(tour)-1):
            p1 = coords[tour[j]]
            p2 = coords[tour[j+1]]
            dist = np.linalg.norm(p1 - p2)
            edge_lengths.append(dist)
            
        max_edges.append(np.max(edge_lengths))
        
    return np.array(max_edges)

def evaluate_metrics(model, data):
    with torch.no_grad():
        cost, _, pi = model(data.to(DEVICE), return_pi=True)
    
    # 1. 基础 Cost
    cost = cost.cpu().numpy()
    
    # 2. Bottleneck Cost (Max Edge)
    bottleneck = get_max_edge_length(data, pi)
    
    return cost, bottleneck

def test_noise_stability(model, base_data):
    """测试抗噪稳定性"""
    # base_data: (1, N, 2)
    # 复制 N 份
    batch = base_data.repeat(NUM_NOISE_TRIALS, 1, 1)
    
    # 加入噪声
    noise = torch.randn_like(batch) * NOISE_LEVEL
    noisy_batch = torch.clamp(batch + noise, 0.0, 1.0)
    
    with torch.no_grad():
        costs, _ = model(noisy_batch.to(DEVICE))
        
    return costs.std().item()

def main():
    torch.manual_seed(SEED)
    print("Loading models...")
    try:
        model_old = load_model(OLD_MODEL_PATH, OLD_ARGS_PATH, ModelOld)
        model_new = load_model(NEW_MODEL_PATH, NEW_ARGS_PATH, ModelNew)
    except Exception as e:
        print(e)
        return

    # 生成测试数据 (混合使用 Uniform 和 Clustered，看看综合效果)
    print(f"\nGenerating {NUM_SAMPLES} samples (50% Uniform, 50% Clustered)...")
    data_u = torch.rand(NUM_SAMPLES // 2, GRAPH_SIZE, 2)
    
    # 生成一些 Cluster 数据
    centers = torch.rand(NUM_SAMPLES // 2, 3, 2)
    batch_indices = torch.arange(NUM_SAMPLES // 2).view(-1, 1).expand(-1, GRAPH_SIZE)
    cluster_indices = torch.randint(0, 3, (NUM_SAMPLES // 2, GRAPH_SIZE))
    data_c = centers[batch_indices, cluster_indices] + torch.randn(NUM_SAMPLES // 2, GRAPH_SIZE, 2) * 0.05
    data_c = torch.clamp(data_c, 0.0, 1.0)
    
    data = torch.cat([data_u, data_c], dim=0)

    # --- 1. 测试 Bottleneck 指标 (Min-Max Edge) ---
    print("\n[Test 1] Bottleneck TSP (Minimizing the longest edge)")
    cost_old, bn_old = evaluate_metrics(model_old, data)
    cost_new, bn_new = evaluate_metrics(model_new, data)
    
    print(f"{'Metric':<20} | {'Old Mean':<10} | {'New Mean':<10} | {'Gap (%)':<10}")
    print("-" * 60)
    print(f"{'Total Cost (Sum)':<20} | {cost_old.mean():<10.4f} | {cost_new.mean():<10.4f} | {(cost_new.mean()-cost_old.mean())/cost_old.mean()*100:<10.2f}")
    print(f"{'Bottleneck (Max)':<20} | {bn_old.mean():<10.4f} | {bn_new.mean():<10.4f} | {(bn_new.mean()-bn_old.mean())/bn_old.mean()*100:<10.2f}")
    
    if bn_new.mean() < bn_old.mean():
        print(">> CONCLUSION: New Method is better at avoiding extremely long edges.")
    
    # --- 2. 测试抗噪稳定性 ---
    print(f"\n[Test 2] Noise Stability (Noise Level={NOISE_LEVEL}, Trials={NUM_NOISE_TRIALS})")
    
    old_stds = []
    new_stds = []
    
    for i in range(NUM_SAMPLES):
        sample = data[i:i+1]
        old_stds.append(test_noise_stability(model_old, sample))
        new_stds.append(test_noise_stability(model_new, sample))
        
        if (i+1) % 20 == 0: print(f"  Processed {i+1}/{NUM_SAMPLES}...")
        
    avg_std_old = np.mean(old_stds)
    avg_std_new = np.mean(new_stds)
    
    print(f"\nAverage Cost Std Dev (Lower is more stable):")
    print(f"Old Model: {avg_std_old:.5f}")
    print(f"New Model: {avg_std_new:.5f}")
    
    if avg_std_new < avg_std_old:
        print(">> CONCLUSION: New Method is MORE ROBUST to sensor noise.")
        
    # 画个图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.boxplot([bn_old, bn_new], labels=['Old', 'New'])
    plt.title('Max Edge Length (Bottleneck)')
    plt.ylabel('Length')
    
    plt.subplot(1, 2, 2)
    plt.boxplot([old_stds, new_stds], labels=['Old', 'New'])
    plt.title(f'Stability (Std Dev under Noise)')
    plt.ylabel('Std Dev')
    
    plt.tight_layout()
    plt.savefig('special_metrics.png')
    print("\nPlot saved to 'special_metrics.png'")

if __name__ == "__main__":
    main()