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

# 我们要测试的规模列表
# 训练是在 20 上，我们要测 20, 50, 甚至 100
TEST_SIZES = [50, 100, 200] 
NUM_SAMPLES = 100        # 每个规模测多少个样本 (为了速度设小一点，正式跑可以用1000)
BATCH_SIZE = 50          # 大图显存占用大，Batch调小
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
    
    # 关键：模型初始化时 graph_size 并不重要，因为 Attention 是动态的
    # 但我们需要确保 embedding_dim 等参数一致
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

def evaluate_on_size(model, size):
    """在特定大小的图上进行测试"""
    # 动态生成不同大小的数据
    data = torch.rand(NUM_SAMPLES, size, 2)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    costs = []
    with torch.no_grad():
        for batch, in loader:
            batch = batch.to(DEVICE)
            cost, _ = model(batch)
            costs.append(cost)
            
    return torch.cat(costs)

def main():
    torch.manual_seed(SEED)
    
    print("Loading models (Trained on Size 20)...")
    try:
        model_old = load_model(OLD_MODEL_PATH, OLD_ARGS_PATH, ModelOld)
        model_new = load_model(NEW_MODEL_PATH, NEW_ARGS_PATH, ModelNew)
    except Exception as e:
        print(e)
        return

    print(f"\nEvaluating Zero-Shot Generalization (Train=20 -> Test={TEST_SIZES})...")
    
    results_old = []
    results_new = []
    gaps = []

    print(f"\n{'Test Size':<10} | {'Old Mean':<12} | {'New Mean':<12} | {'Gap (%)':<10}")
    print("-" * 50)

    for size in TEST_SIZES:
        try:
            cost_old = evaluate_on_size(model_old, size)
            cost_new = evaluate_on_size(model_new, size)
            
            mean_old = cost_old.mean().item()
            mean_new = cost_new.mean().item()
            
            # Gap: 负数表示新模型更好
            gap = (mean_new - mean_old) / mean_old * 100
            
            results_old.append(mean_old)
            results_new.append(mean_new)
            gaps.append(gap)
            
            print(f"{size:<10} | {mean_old:<12.4f} | {mean_new:<12.4f} | {gap:<10.2f}")
            
        except RuntimeError as e:
            print(f"{size:<10} | OOM or Error: {e}")
            results_old.append(0)
            results_new.append(0)
            gaps.append(0)

    # --- 可视化泛化趋势 ---
    plt.figure(figsize=(10, 6))
    
    # 我们画 "Gap 随 Size 的变化趋势"
    # 如果曲线向下倾斜（越来越负），说明规模越大，新模型优势越明显
    plt.plot(TEST_SIZES, gaps, marker='o', linewidth=2, label='Gap (New - Old) %')
    plt.axhline(0, color='gray', linestyle='--')
    
    plt.title('Generalization Capability: Performance Gap vs Graph Size')
    plt.xlabel('Test Graph Size (Nodes)')
    plt.ylabel('Gap % (Lower is Better for New Model)')
    plt.grid(True)
    plt.legend()
    
    plt.savefig('generalization_trend.png', dpi=300)
    print("\nTrend plot saved to 'generalization_trend.png'")
    
    # 简单的文字结论
    if len(gaps) > 1 and gaps[-1] < gaps[0]:
        print("\n>> CONCLUSION: The New Method generalizes BETTER to larger graphs.")
    elif len(gaps) > 1 and gaps[-1] > gaps[0]:
        print("\n>> CONCLUSION: The Old Method generalizes better.")
    else:
        print("\n>> CONCLUSION: Generalization capability is similar.")

if __name__ == "__main__":
    main()