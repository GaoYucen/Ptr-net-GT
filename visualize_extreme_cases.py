import os
import io
import torch
import matplotlib.pyplot as plt
from PIL import Image
from utils import torch_load_cpu, load_problem

# 引入您的模型定义
from nets.attention_model_dual_no_context import AttentionModel

def generate_cluster_data(num_nodes, num_clusters=3, std=0.05):
    """生成强聚类数据"""
    centers = torch.rand(1, num_clusters, 2)
    nodes_per_cluster = num_nodes // num_clusters
    parts = []
    
    for i in range(num_clusters):
        # 在中心点周围生成高斯分布的点
        pts = centers[:, i:i+1, :] + torch.randn(1, nodes_per_cluster, 2) * std
        parts.append(pts)
    
    # 处理剩余节点
    remainder = num_nodes - nodes_per_cluster * num_clusters
    if remainder > 0:
        parts.append(torch.rand(1, remainder, 2))
        
    data = torch.cat(parts, dim=1)
    return data.clamp(0, 1) # 截断在 [0,1] 范围内

def generate_gap_data(num_nodes):
    """生成被鸿沟隔开的两堆数据 (左右分布)"""
    half = num_nodes // 2
    
    # 左边部分 x: [0, 0.4]
    left = torch.rand(1, half, 2)
    left[:, :, 0] *= 0.4 
    
    # 右边部分 x: [0.6, 1.0] (中间空出 0.2 的 Gap)
    right = torch.rand(1, num_nodes - half, 2)
    right[:, :, 0] = right[:, :, 0] * 0.4 + 0.6 
    
    data = torch.cat([left, right], dim=1)
    return data

def visualize_trajectory(model, data, title_prefix, save_name):
    """运行模型并生成 GIF，带跳跃检测"""
    
    opts = model.opts # 获取保存的 opts
    
    # 1. 运行模型
    model.set_decode_type("greedy")
    model.eval()
    with torch.no_grad():
        data = data.to(opts.device)
        cost, log_likelihood, pi = model(data, return_pi=True)

    # 2. 解析数据
    locs = data.cpu().numpy()[0] 
    seq = pi.cpu().numpy()[0]
    
    actions = []
    for k in range(0, len(seq), 2):
        actions.append((seq[k], seq[k+1]))

    print(f"[{title_prefix}] 步数: {len(actions)}, Cost: {cost.item():.4f}")

    # 3. 绘图循环
    frames = []
    current_edges = []
    
    x_min, x_max = 0, 1
    y_min, y_max = 0, 1
    
    for step, (u, v) in enumerate(actions):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(x_min - 0.05, x_max + 0.05)
        ax.set_ylim(y_min - 0.05, y_max + 0.05)
        ax.set_aspect('equal')
        
        # 画点
        ax.scatter(locs[:, 0], locs[:, 1], c='blue', s=50, zorder=5)
        
        # 画已完成的边
        for (pu, pv) in current_edges:
            p1, p2 = locs[pu], locs[pv]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='gray', alpha=0.5, lw=2)
            
        # --- 核心：检测并绘制 JUMP ---
        is_jump = False
        if step > 0:
            prev_head = actions[step-1][1]
            # 如果当前起点 != 上一步终点，说明发生了跳跃
            if u != prev_head:
                is_jump = True
                p_prev = locs[prev_head]
                p_curr = locs[u]
                # 画蓝色虚线表示跳跃逻辑（并非真实的边）
                ax.plot([p_prev[0], p_curr[0]], [p_prev[1], p_curr[1]], 
                        c='blue', linestyle='--', alpha=0.6, lw=1.5, label='JUMP (Logic)')
                ax.text((p_prev[0]+p_curr[0])/2, (p_prev[1]+p_curr[1])/2, "JUMP!", 
                        color='blue', fontsize=10, fontweight='bold')

        # 画当前边
        p_tail, p_head = locs[u], locs[v]
        edge_color = 'magenta' if is_jump else 'red'
        ax.plot([p_tail[0], p_head[0]], [p_tail[1], p_head[1]], c=edge_color, lw=3, label='Current Edge')
        
        # 标记起终点
        ax.scatter(p_tail[0], p_tail[1], c='green', s=150, zorder=6, label='Tail') 
        ax.scatter(p_head[0], p_head[1], c='orange', s=150, zorder=6, label='Head')
        
        # 箭头
        ax.annotate("", xy=(p_head[0], p_head[1]), xytext=(p_tail[0], p_tail[1]),
                    arrowprops=dict(arrowstyle="->", color=edge_color, lw=2))
        
        ax.set_title(f"{title_prefix} | Step {step+1}", fontsize=12)
        if step == 0 or is_jump:
            ax.legend(loc='upper right')
        
        ax.axis('off')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)
        
        current_edges.append((u, v))

    for _ in range(15):
        frames.append(frames[-1])

    frames[0].save(save_name, save_all=True, append_images=frames[1:], duration=600, loop=0)
    print(f"  >>> 生成成功: {save_name}")

if __name__ == "__main__":
    from options import get_options
    
    # 1. 配置参数
    opts = get_options()
    opts.use_cuda = torch.cuda.is_available()
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    
    if opts.load_path is None:
        opts.load_path = r"E:\code\attention-learn-to-route-master 20.43.21\attention-learn-to-route-master\outputs\tsp_20\tsp20_dual_warmup_20260123T160816\best-model.pt"
    
    print(f"正在加载模型: {opts.load_path}")
    state = torch_load_cpu(opts.load_path)
    
    # 2. 初始化模型
    problem = load_problem('tsp')
    model = AttentionModel(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size
    ).to(opts.device)
    
    # 处理 DataParallel 权重键名
    model_state_dict = state.get('model', {})
    if list(model_state_dict.keys())[0].startswith('module.'):
        model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict)
    
    # 将 opts 挂载到 model 上方便调用
    model.opts = opts

    # 3. 生成并运行特殊案例
    N = 50 # 节点数
    
    print("\n--- Case 1: 强聚类 (3 Clusters) ---")
    data_cluster = generate_cluster_data(N, num_clusters=3)
    visualize_trajectory(model, data_cluster, "Cluster Mode", "extreme_cluster.gif")
    
    print("\n--- Case 2: 断裂分布 (Gap) ---")
    data_gap = generate_gap_data(N)
    visualize_trajectory(model, data_gap, "Gap Mode", "extreme_gap.gif")