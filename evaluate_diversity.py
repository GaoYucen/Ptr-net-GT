import os
import json
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from utils import load_problem

# ================= 配置区域 =================



OLD_MODEL_PATH = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356\epoch-26.pt'
OLD_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356\args.json'  # 必须指定对应的 args.json

NEW_MODEL_PATH = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\best-model_91.pt'
NEW_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\args.json'  # 必须指定对应的 args.json

GRAPH_SIZE = 50
NUM_INSTANCES = 10     # 测试多少个图
SAMPLE_SIZE = 3000      # 每个图采样多少次 (探索次数)
BATCH_SIZE = 100
SEED = 333
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
    # 【关键】设置为 sampling 模式，开启随机探索
    model.set_decode_type("sampling")
    return model

def evaluate_diversity(model, data):
    """
    对 data (1, N, 2) 进行 SAMPLE_SIZE 次采样
    返回: 最佳Cost, 平均Cost, 唯一解的数量
    """
    # 构造 batch: 复制 SAMPLE_SIZE 份
    batch_data = data.repeat(SAMPLE_SIZE, 1, 1)
    
    with torch.no_grad():
        costs, _, pi = model(batch_data.to(DEVICE), return_pi=True)
    
    costs = costs.cpu().numpy()
    pi = pi.cpu().numpy()
    
    best_cost = np.min(costs)
    avg_cost = np.mean(costs)
    
    # 计算多样性：有多少个不重复的路径？
    # 将路径转为 tuple 以便 hash，注意 TSP 路径的循环同构问题这里简化处理，直接比较序列
    # (更严谨的话应该把路径 normalize 成从 0 开始的序列再比较，但这里足以说明问题)
    unique_paths = set(tuple(p) for p in pi)
    diversity_ratio = len(unique_paths) / SAMPLE_SIZE * 100
    
    return best_cost, avg_cost, diversity_ratio

def main():
    torch.manual_seed(SEED)
    
    print("Loading models...")
    try:
        model_old = load_model(OLD_MODEL_PATH, OLD_ARGS_PATH, ModelOld)
        model_new = load_model(NEW_MODEL_PATH, NEW_ARGS_PATH, ModelNew)
    except Exception as e:
        print(e)
        return

    print(f"\nEvaluating Exploration Capability (Sampling N={SAMPLE_SIZE})...")
    
    results = []
    
    print(f"\n{'ID':<3} | {'Old Best':<10} | {'New Best':<10} | {'Gap (%)':<8} | {'Old Div%':<8} | {'New Div%':<8}")
    print("-" * 70)
    
    wins = 0
    ties = 0
    
    for i in range(NUM_INSTANCES):
        data = torch.rand(1, GRAPH_SIZE, 2)
        
        old_best, old_avg, old_div = evaluate_diversity(model_old, data)
        new_best, new_avg, new_div = evaluate_diversity(model_new, data)
        
        gap = (new_best - old_best) / old_best * 100
        
        if new_best < old_best: wins += 1
        if abs(new_best - old_best) < 1e-5: ties += 1
            
        print(f"{i:<3} | {old_best:<10.4f} | {new_best:<10.4f} | {gap:<8.2f} | {old_div:<8.0f} | {new_div:<8.0f}")
        
        results.append({
            'gap': gap,
            'new_div': new_div,
            'old_div': old_div
        })

    print("-" * 70)
    print(f"Summary over {NUM_INSTANCES} instances:")
    print(f"New Model Wins: {wins}")
    print(f"Ties: {ties}")
    print(f"Old Model Wins: {NUM_INSTANCES - wins - ties}")
    
    avg_gap = np.mean([r['gap'] for r in results])
    avg_old_div = np.mean([r['old_div'] for r in results])
    avg_new_div = np.mean([r['new_div'] for r in results])
    
    print(f"\nAverage Gap (Best Sampled): {avg_gap:.2f}% (Negative means New Model found better solution)")
    print(f"Average Diversity (Unique%): Old={avg_old_div:.1f}%, New={avg_new_div:.1f}%")

    if avg_new_div > avg_old_div:
        print("\n>> CONCLUSION: New Method explores the solution space MORE broadly.")
    else:
        print("\n>> CONCLUSION: Old Method explores more broadly.")

if __name__ == "__main__":
    main()