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

# 2. 测试配置
# 测试的规模列表：从小到大
TEST_SIZES = [50, 100, 200] 

# 评估设置
NUM_INSTANCES = 50      # 每个规模测试多少个不同的图 (Instance)
SAMPLE_WIDTH = 100      # 【关键】每个图采样多少次 (取 Best of 100)
BATCH_SIZE = 10         # 为了防止显存溢出，这里Batch要小，因为实际 tensor 是 BATCH * SAMPLE_WIDTH
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
    """修复旧模型权重键名不匹配问题"""
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
    
    # 【核心设置】开启采样模式
    model.set_decode_type("sampling")
    model.eval()
    return model

def evaluate_best_of_n(model, size):
    """
    在特定 Size 上进行 Best-of-N 评估
    """
    # 生成 NUM_INSTANCES 个不同的图
    # data shape: (N, Size, 2)
    data = torch.rand(NUM_INSTANCES, size, 2)
    
    # 我们需要逐个(或小Batch)处理这些 Instance，对每个 Instance 进行 SAMPLE_WIDTH 次扩充
    # 这里为了代码简单和安全，我们逐个 Instance 处理
    
    best_costs = []
    
    with torch.no_grad():
        for i in range(NUM_INSTANCES):
            # 取出 1 个图: (1, Size, 2)
            instance = data[i:i+1]
            
            # 扩充成 (SAMPLE_WIDTH, Size, 2)
            # 这就相当于让模型对这同一个图跑 SAMPLE_WIDTH 遍
            batch = instance.repeat(SAMPLE_WIDTH, 1, 1).to(DEVICE)
            
            # 运行模型
            costs, _ = model(batch)
            
            # 取最小的 Cost (Best of N)
            best_cost = costs.min().item()
            best_costs.append(best_cost)
            
            # 简单的进度打印
            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{NUM_INSTANCES} instances...")

    # 返回所有最佳 Cost 的 Tensor
    return torch.tensor(best_costs)

def main():
    torch.manual_seed(SEED)
    
    print("Loading models (Mode: Sampling)...")
    try:
        model_old = load_model(OLD_MODEL_PATH, OLD_ARGS_PATH, ModelOld)
        model_new = load_model(NEW_MODEL_PATH, NEW_ARGS_PATH, ModelNew)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"\nSTARTING TEST: Generalization + Sampling Exploration")
    print(f"Settings: Train Size=20 | Test Sizes={TEST_SIZES}")
    print(f"Strategy: Generate {NUM_INSTANCES} graphs per size.")
    print(f"          For EACH graph, sample {SAMPLE_WIDTH} solutions and pick the BEST one.")
    
    summary_results = {
        'sizes': [],
        'old_means': [],
        'new_means': [],
        'gaps': []
    }

    print(f"\n{'Size':<6} | {'Old (Best-of-N)':<18} | {'New (Best-of-N)':<18} | {'Gap (%)':<10}")
    print("-" * 60)

    for size in TEST_SIZES:
        try:
            # 评估旧模型
            costs_old = evaluate_best_of_n(model_old, size)
            mean_old = costs_old.mean().item()
            
            # 评估新模型
            costs_new = evaluate_best_of_n(model_new, size)
            mean_new = costs_new.mean().item()
            
            # 计算 Gap
            gap = (mean_new - mean_old) / mean_old * 100
            
            # 记录
            summary_results['sizes'].append(size)
            summary_results['old_means'].append(mean_old)
            summary_results['new_means'].append(mean_new)
            summary_results['gaps'].append(gap)
            
            print(f"{size:<6} | {mean_old:<18.4f} | {mean_new:<18.4f} | {gap:<10.2f}")
            
        except RuntimeError as e:
            print(f"{size:<6} | OOM/Error: {e}")

    # --- 可视化绘图 ---
    plt.figure(figsize=(10, 6))
    
    # 画出 Gap 曲线
    sizes = summary_results['sizes']
    gaps = summary_results['gaps']
    
    plt.plot(sizes, gaps, marker='o', linewidth=3, color='purple', label='Gap (New - Old)')
    plt.axhline(0, color='gray', linestyle='--')
    
    # 标注每个点的值
    for x, y in zip(sizes, gaps):
        plt.text(x, y + 0.5, f"{y:.2f}%", ha='center', fontweight='bold')

    plt.title(f'Generalization w/ Exploration (Best of {SAMPLE_WIDTH})\nGap < 0 means New Model is Better', fontsize=14)
    plt.xlabel('Graph Size (Nodes)', fontsize=12)
    plt.ylabel('Performance Gap %', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    save_name = 'generalization_sampling_trend.png'
    plt.savefig(save_name, dpi=300)
    print(f"\nTrend plot saved to '{save_name}'")
    
    # 结论分析
    if len(gaps) > 0:
        last_gap = gaps[-1]
        print("\n" + "="*50)
        if last_gap < -1.0:
            print(">> MAJOR FINDING: The New Method significantly outperforms the Old Method")
            print("   on large graphs when allowed to explore (Sampling).")
            print("   This suggests a better upper-bound in the solution space.")
        elif last_gap > 1.0:
            print(">> FINDING: The Old Method scales better even with sampling.")
        else:
            print(">> FINDING: Both methods perform similarly.")
        print("="*50)

if __name__ == "__main__":
    main()