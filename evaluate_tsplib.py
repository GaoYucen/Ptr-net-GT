import os
import glob
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import load_problem

# ================= 配置区域 =================
# 1. 模型路径

OLD_MODEL_PATH = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356\epoch-26.pt'
OLD_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_50\tsp20_rollout_20260122T114356\args.json'  # 必须指定对应的 args.json


NEW_MODEL_PATH = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\best-model_91.pt'
NEW_ARGS_PATH  = r'E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\args.json'  # 必须指定对应的 args.json

# 2. TSPLIB 数据文件夹路径
TSPLIB_DIR = 'data/tsplib' 
# 如果你想测试特定的几个文件，可以在这里指定文件名关键词，否则测试目录下所有 .tsp
FILTER_Files = [] # e.g. ['eil51', 'berlin52']

SEED = 1234
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ===========================================

from nets.attention_model import AttentionModel as ModelOld
try:
    from nets.attention_model_dual_crossStep import AttentionModel as ModelNew
except ImportError:
    from nets.attention_model import AttentionModel as ModelNew

# --- 1. TSPLIB 解析器 ---
def read_tsplib(filepath):
    """
    读取 .tsp 文件，提取坐标并归一化到 [0, 1]
    返回: 
      coords_norm: 归一化后的 tensor (1, N, 2)
      coords_raw: 原始坐标 numpy array (N, 2) 用于画图
      name: 实例名称
    """
    coords = []
    name = os.path.basename(filepath).split('.')[0]
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
        # 寻找坐标数据起始点
        start_reading = False
        for line in lines:
            if "NODE_COORD_SECTION" in line:
                start_reading = True
                continue
            if "EOF" in line:
                break
            
            if start_reading:
                parts = line.strip().split()
                # 通常格式: ID x y
                # 有些文件可能是空格分隔，有些是 tab
                if len(parts) >= 3:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        coords.append([x, y])
                    except ValueError:
                        pass
    
    if len(coords) == 0:
        raise ValueError(f"No coordinates found in {filepath}")

    coords_raw = np.array(coords)
    
    # 归一化 (Min-Max Scaling) 到 [0, 1]
    # 模型训练时使用的是 0-1 分布，必须缩放才能有正确表现
    min_val = coords_raw.min(axis=0)
    max_val = coords_raw.max(axis=0)
    scale = max_val - min_val
    # 防止除以0
    scale[scale == 0] = 1.0
    
    coords_norm = (coords_raw - min_val) / scale
    
    # 转为 Tensor: (1, N, 2)
    coords_tensor = torch.tensor(coords_norm, dtype=torch.float32).unsqueeze(0)
    
    return coords_tensor, coords_raw, name

# --- 2. 模型加载 ---
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
    
    # 评估 TSPLIB 这种难问题，建议开启 Sampling 模式以获得最佳性能
    # 如果想看这一眼贪婪解，可以改为 "greedy"
    model.set_decode_type("greedy") 
    model.eval()
    return model

# --- 3. 可视化对比 ---
def plot_tsplib_comparison(name, coords_raw, pi_old, cost_old, pi_new, cost_new):
    """
    coords_raw: 原始的真实坐标 (N, 2)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # 计算 Gap
    gap = (cost_new - cost_old) / cost_old * 100
    gap_str = f"{gap:.2f}%"
    gap_color = 'green' if gap < 0 else 'red'

    # --- Old Model ---
    ax = axes[0]
    ax.scatter(coords_raw[:,0], coords_raw[:,1], c='blue', s=20)
    ax.scatter(coords_raw[0,0], coords_raw[0,1], c='red', marker='*', s=150, label='Depot')
    
    tour = list(pi_old)
    if tour[-1] != tour[0]: tour.append(tour[0])
    
    for j in range(len(tour)-1):
        p1, p2 = coords_raw[tour[j]], coords_raw[tour[j+1]]
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], c='gray', alpha=0.7)
        
    ax.set_title(f"Old Method\nNorm Cost: {cost_old:.4f}", fontsize=14)
    ax.axis('off')

    # --- New Model ---
    ax = axes[1]
    ax.scatter(coords_raw[:,0], coords_raw[:,1], c='blue', s=20)
    ax.scatter(coords_raw[0,0], coords_raw[0,1], c='red', marker='*', s=150)
    
    # New model outputs Edges
    tails, heads = pi_new[0::2], pi_new[1::2]
    for t, h in zip(tails, heads):
        p1, p2 = coords_raw[t], coords_raw[h]
        ax.plot([p1[0],p2[0]], [p1[1],p2[1]], c='black', alpha=0.9, linewidth=1.5)
        
    ax.set_title(f"New Method\nNorm Cost: {cost_new:.4f} (Gap: {gap_str})", fontsize=14, color=gap_color)
    ax.axis('off')

    plt.suptitle(f"TSPLIB Instance: {name} (N={len(coords_raw)})", fontsize=16)
    
    save_dir = 'vis_tsplib'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{name}.png", dpi=300)
    plt.close()

def main():
    torch.manual_seed(SEED)
    
    # 查找所有 .tsp 文件
    tsp_files = glob.glob(os.path.join(TSPLIB_DIR, "*.tsp"))
    if FILTER_Files:
        tsp_files = [f for f in tsp_files if any(x in f for x in FILTER_Files)]
    
    if not tsp_files:
        print(f"No .tsp files found in {TSPLIB_DIR}. Please add some files.")
        print("Download from: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/")
        return

    print("Loading models...")
    model_old = load_model(OLD_MODEL_PATH, OLD_ARGS_PATH, ModelOld)
    model_new = load_model(NEW_MODEL_PATH, NEW_ARGS_PATH, ModelNew)

    print(f"\nEvaluating on {len(tsp_files)} TSPLIB instances...")
    print(f"{'Instance':<15} | {'Size':<5} | {'Old (Norm)':<12} | {'New (Norm)':<12} | {'Gap (%)':<8}")
    print("-" * 65)

    for filepath in tsp_files:
        try:
            # 1. 读取并处理数据
            coords_tensor, coords_raw, name = read_tsplib(filepath)
            N = len(coords_raw)
            
            # 2. 推理
            with torch.no_grad():
                # Old Model
                cost_old, _, pi_old = model_old(coords_tensor.to(DEVICE), return_pi=True)
                # New Model
                cost_new, _, pi_new = model_new(coords_tensor.to(DEVICE), return_pi=True)
            
            cost_old = cost_old.item()
            cost_new = cost_new.item()
            pi_old = pi_old.cpu().numpy()[0]
            pi_new = pi_new.cpu().numpy()[0]
            
            # 3. 统计
            gap = (cost_new - cost_old) / cost_old * 100
            
            print(f"{name:<15} | {N:<5} | {cost_old:<12.4f} | {cost_new:<12.4f} | {gap:<8.2f}")
            
            # 4. 画图
            plot_tsplib_comparison(name, coords_raw, pi_old, cost_old, pi_new, cost_new)
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print("-" * 65)
    print(f"Visualization results saved to 'vis_tsplib/' folder.")

if __name__ == "__main__":
    main()