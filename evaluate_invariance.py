import os
import json
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from utils import load_problem

# ================= 配置区域 =================

OLD_MODEL_PATH = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356\epoch-49.pt'
OLD_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356\args.json'  # 必须指定对应的 args.json


NEW_MODEL_PATH = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\best-model.pt'
NEW_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\args.json'  # 必须指定对应的 args.json

GRAPH_SIZE = 50
NUM_TEST_INSTANCES = 50   # 测试多少个不同的图
NUM_PERMUTATIONS = 20     # 每个图打乱多少次顺序
BATCH_SIZE = 100
SEED = 1234
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ===========================================

from nets.attention_model import AttentionModel as ModelOld
try:
    from nets.attention_model_dual_crossStep import AttentionModel as ModelNew
except ImportError:
    from nets.attention_model import AttentionModel as ModelNew

# --- 辅助函数 ---
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

def test_permutation_stability(model, base_data):
    """
    对 base_data (1, N, 2) 进行多次循环移位，测试模型的输出稳定性
    """
    # 构造 batch：复制 base_data NUM_PERMUTATIONS 次
    # (B, N, 2)
    batch_data = base_data.repeat(NUM_PERMUTATIONS, 1, 1)
    
    # 对每一个样本进行不同程度的 roll (循环移位)
    # 这模拟了：同一个图，但是输入的节点顺序变了（比如节点0变成了列表里的第5个）
    for i in range(NUM_PERMUTATIONS):
        batch_data[i] = torch.roll(batch_data[i], shifts=i, dims=0)
        
    with torch.no_grad():
        costs, _, _ = model(batch_data.to(DEVICE), return_pi=True)
        
    costs = costs.cpu().numpy()
    
    # 计算指标
    mean_cost = np.mean(costs)
    std_cost = np.std(costs) # 标准差越小越好
    gap_max_min = np.max(costs) - np.min(costs) # 极差
    
    return mean_cost, std_cost, gap_max_min

def main():
    torch.manual_seed(SEED)
    
    print("Loading models...")
    try:
        model_old = load_model(OLD_MODEL_PATH, OLD_ARGS_PATH, ModelOld)
        model_new = load_model(NEW_MODEL_PATH, NEW_ARGS_PATH, ModelNew)
    except Exception as e:
        print(e)
        return

    print(f"\nTesting Permutation Invariance on {NUM_TEST_INSTANCES} instances...")
    print(f"Each instance is permuted {NUM_PERMUTATIONS} times.")
    
    # 存储统计结果
    old_stds = []
    new_stds = []
    old_means = []
    new_means = []

    print(f"\n{'Instance ID':<12} | {'Old Std':<10} | {'New Std':<10} | {'Winner':<10}")
    print("-" * 50)

    for i in range(NUM_TEST_INSTANCES):
        # 生成一个随机图 (1, N, 2)
        base_data = torch.rand(1, GRAPH_SIZE, 2)
        
        # 测试旧模型
        mean_old, std_old, _ = test_permutation_stability(model_old, base_data)
        # 测试新模型
        mean_new, std_new, _ = test_permutation_stability(model_new, base_data)
        
        old_stds.append(std_old)
        new_stds.append(std_new)
        old_means.append(mean_old)
        new_means.append(mean_new)
        
        winner = "NEW" if std_new < std_old else "OLD"
        if i < 10: # 只打印前10个的详细信息
            print(f"{i:<12} | {std_old:<10.4f} | {std_new:<10.4f} | {winner:<10}")

    # --- 总体统计 ---
    avg_std_old = np.mean(old_stds)
    avg_std_new = np.mean(new_stds)
    
    avg_mean_old = np.mean(old_means)
    avg_mean_new = np.mean(new_means)

    print("\n" + "="*50)
    print("FINAL RESULTS SUMMARY")
    print("="*50)
    print(f"Metric              | Old Model  | New Model")
    print("-" * 50)
    print(f"Avg Cost (Performance)| {avg_mean_old:<10.4f} | {avg_mean_new:<10.4f}")
    print(f"Avg Std Dev (Stability)| {avg_std_old:<10.4f} | {avg_std_new:<10.4f}")
    print("-" * 50)
    
    # 判定
    improvement = (avg_std_old - avg_std_new) / avg_std_old * 100
    print(f"Stability Improvement: {improvement:.2f}%")
    if avg_std_new < avg_std_old:
        print(">> CONCLUSION: The New Method is MORE ROBUST to input permutations.")
    else:
        print(">> CONCLUSION: The Old Method is more stable.")

    # --- 可视化波动 ---
    plt.figure(figsize=(10, 6))
    plt.boxplot([old_stds, new_stds], labels=['Old Model', 'New Model'])
    plt.title(f'Standard Deviation of Costs under Input Permutation\n(Lower is Better/More Robust)')
    plt.ylabel('Cost Std Dev')
    plt.grid(True, axis='y', alpha=0.5)
    plt.savefig('robustness_comparison.png', dpi=300)
    print("\nBoxplot saved to 'robustness_comparison.png'")

if __name__ == "__main__":
    main()