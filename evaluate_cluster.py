import os
import json
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from utils import load_problem

# ================= 配置区域 =================

# 1. 目录配置 (注意：这里只需配置到文件夹路径，不要包含具体的文件名)
OLD_MODEL_DIR = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356'
NEW_MODEL_DIR = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\tsp20_dual_warmup_20260122T064518.output'

# 参数文件路径
OLD_ARGS_PATH  = os.path.join(OLD_MODEL_DIR, 'args.json')
NEW_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\args.json'

# 2. Epoch 范围配置
OLD_EPOCH_RANGE = range(0, 48)  # [20, 21, ..., 49]
NEW_EPOCH_RANGE = range(51, 99) # [70, 71, ..., 99]

# 3. 测试配置
GRAPH_SIZE = 50           
NUM_TEST_SAMPLES = 3000   
BATCH_SIZE = 256
SEED = 1234
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 分布类型列表
DISTRIBUTIONS = ['clustered'] 
# ===========================================

# 动态导入模型
from nets.attention_model import AttentionModel as ModelOld
try:
    from nets.attention_model_dual_crossStep import AttentionModel as ModelNew
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
    if not os.path.exists(model_path): 
        return None # 如果文件不存在返回 None
    
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
    
    with torch.no_grad():
        for i, (batch,) in enumerate(loader):
            batch = batch.to(DEVICE)
            cost, _ = model(batch, return_pi=False) # 这里不需要 pi，为了速度只返回 cost
            costs.append(cost)
            
    return torch.cat(costs)

def main():
    torch.manual_seed(SEED)
    
    # 检查 Epoch 数量是否对齐
    if len(OLD_EPOCH_RANGE) != len(NEW_EPOCH_RANGE):
        print(f"Warning: Number of epochs do not match! Old: {len(OLD_EPOCH_RANGE)}, New: {len(NEW_EPOCH_RANGE)}")
        # 取最小长度以防报错
        length = min(len(OLD_EPOCH_RANGE), len(NEW_EPOCH_RANGE))
    else:
        length = len(OLD_EPOCH_RANGE)

    print(f"Starting Batch Evaluation...")
    print(f"Old Model Dir: {OLD_MODEL_DIR}")
    print(f"New Model Dir: {NEW_MODEL_DIR}")
    print(f"Comparing {length} pairs of checkpoints.\n")

    # 1. 循环测试分布 (通常只有一个 'clustered')
    for dist_name in DISTRIBUTIONS:
        print(f"Generating Data for distribution: {dist_name} (Size: {NUM_TEST_SAMPLES})...")
        # --- 关键：在循环外生成一次数据，保证所有 Epoch 对比的是同一组数据 ---
        data = generate_data(dist_name, NUM_TEST_SAMPLES, GRAPH_SIZE)
        
        # 打印漂亮的表头
        header = (f"{'Index':<5} | "
                  f"{'Old Epoch':<9} | {'Old Cost':<10} | "
                  f"{'New Epoch':<9} | {'New Cost':<10} | "
                  f"{'Gap (%)':<8}")
        
        print(f"\nResults for {dist_name}:")
        print(f"{'='*len(header)}")
        print(header)
        print(f"{'-'*len(header)}")

        # 2. 循环 Epoch
        for i in range(length):
            old_ep = OLD_EPOCH_RANGE[i]
            new_ep = NEW_EPOCH_RANGE[i]

            # 构造文件名
            old_path = os.path.join(OLD_MODEL_DIR, f'epoch-{old_ep}.pt')
            new_path = os.path.join(NEW_MODEL_DIR, f'epoch-{new_ep}.pt')

            # 加载模型 (Old)
            model_old = load_model(old_path, OLD_ARGS_PATH, ModelOld)
            if model_old is None:
                print(f"{i:<5} | {old_ep:<9} | {'MISSING':<10} | ...")
                continue

            # 加载模型 (New)
            model_new = load_model(new_path, NEW_ARGS_PATH, ModelNew)
            if model_new is None:
                print(f"{i:<5} | {old_ep:<9} | {'Done':<10} | {new_ep:<9} | {'MISSING':<10} | ...")
                continue

            # 评估
            costs_old = evaluate(model_old, data)
            costs_new = evaluate(model_new, data)

            # 统计
            mean_old = costs_old.mean().item()
            mean_new = costs_new.mean().item()
            gap = ((mean_new - mean_old) / mean_old) * 100

            # 打印单行结果
            print(f"{i:<5} | "
                  f"{old_ep:<9} | {mean_old:<10.4f} | "
                  f"{new_ep:<9} | {mean_new:<10.4f} | "
                  f"{gap:<8.2f}")

    print(f"{'='*len(header)}")
    print("Evaluation Complete.")

if __name__ == "__main__":
    main()