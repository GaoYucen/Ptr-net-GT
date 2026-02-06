import os
import io
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from utils import load_problem

# 引入您的模型定义
from nets.attention_model_dual_crossStep import AttentionModel

def visualize_trajectory(model, opts, save_name="planning_process.gif"):
    """
    运行模型并生成路径规划过程的 GIF
    """
    # 1. 准备数据 (生成一个随机样本)
    problem = load_problem(opts.problem)
    # 生成 1 个样本，大小为 opts.graph_size
    dataset = problem.make_dataset(
        size=opts.graph_size, 
        num_samples=1, 
        distribution=opts.data_distribution
    )
    dataloader = DataLoader(dataset, batch_size=1)
    
    # 获取一个 Batch
    batch = next(iter(dataloader))
    
    # 2. 运行模型获取决策序列 (pi)
    # --- 修正点开始 ---
    model.set_decode_type("greedy") # 必须显式告诉模型使用贪婪策略
    # --- 修正点结束 ---
    
    model.eval()
    with torch.no_grad():
        # 移动到设备
        batch = batch.to(opts.device)
        
        # 关键：设置 return_pi=True 以获取动作序列
        cost, log_likelihood, pi = model(batch, return_pi=True)

    # 3. 解析序列与坐标
    # batch (Tensor): (1, N, 2)
    locs = batch.cpu().numpy()[0] 
    seq = pi.cpu().numpy()[0]
    
    # 将序列解析为 (Tail, Head) 对
    actions = []
    for k in range(0, len(seq), 2):
        tail_idx = seq[k]
        head_idx = seq[k+1]
        actions.append((tail_idx, head_idx))
        
    print(f"生成的路径总步数: {len(actions)}")
    print(f"最终 Cost: {cost.item():.4f}")

    # 4. 生成 GIF 帧
    frames = []
    current_edges = []
    
    # 预先设置画布范围
    x_min, x_max = locs[:, 0].min(), locs[:, 0].max()
    y_min, y_max = locs[:, 1].min(), locs[:, 1].max()
    margin = 0.05
    
    for step, (u, v) in enumerate(actions):
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_aspect('equal')
        
        # 绘制所有节点
        ax.scatter(locs[:, 0], locs[:, 1], c='blue', s=50, zorder=5, label='Nodes')
        
        # 绘制已完成的边 (灰色)
        for (pu, pv) in current_edges:
            p1, p2 = locs[pu], locs[pv]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], c='gray', alpha=0.5, lw=2)
            
        # 绘制当前正在生成的边 (红色箭头)
        p_tail, p_head = locs[u], locs[v]
        ax.plot([p_tail[0], p_head[0]], [p_tail[1], p_head[1]], c='red', lw=3, label='Current Step')
        
        # 特别标注起终点
        ax.scatter(p_tail[0], p_tail[1], c='green', s=150, zorder=6, label='Tail (Start)') 
        ax.scatter(p_head[0], p_head[1], c='orange', s=150, zorder=6, label='Head (End)')
        
        # 画箭头指示方向
        ax.annotate("", xy=(p_head[0], p_head[1]), xytext=(p_tail[0], p_tail[1]),
                    arrowprops=dict(arrowstyle="->", color='red', lw=2))
        
        # 标题与图例
        ax.set_title(f"Step {step+1}/{len(actions)}\nTail: {u} -> Head: {v}", fontsize=12)
        if step == 0: 
            ax.legend(loc='upper right')
        
        ax.axis('off')

        # 保存帧到内存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)
        
        current_edges.append((u, v))

    # 最后一帧停留一会儿
    for _ in range(10):
        frames.append(frames[-1])

    frames[0].save(save_name, save_all=True, append_images=frames[1:], duration=500, loop=0)
    print(f"可视化已保存为: {save_name}")

if __name__ == "__main__":
    import argparse
    from options import get_options # 复用您现有的 options
    from utils import torch_load_cpu

    # 获取基础参数
    opts = get_options()
    
    # 这里可以手动覆盖一些参数用于测试
    opts.graph_size = 20    # 比如测试 20 个点
    opts.batch_size = 1
    opts.use_cuda = torch.cuda.is_available()
    opts.device = torch.device("cuda" if opts.use_cuda else "cpu")
    
    # 模型路径 (请修改为您实际的 .pt 文件路径)
    # MODEL_PATH = "outputs/tsp_20/run_name/epoch-99.pt"
    # 如果您想直接在命令行指定: python visualize.py --load_path ...
    
    if opts.load_path is None and opts.resume is None:
        raise ValueError("请通过 --load_path 指定模型路径 (例如: --load_path 'outputs/tsp_20/best-model.pt')")
        
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    print(f"正在加载模型: {load_path}")
    
    # 加载模型权重
    state = torch_load_cpu(load_path)
    
    # 初始化模型
    problem = load_problem(opts.problem)
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
    
    # 加载参数 (处理可能的 DataParallel 前缀)
    model_state_dict = state.get('model', {})
    # 如果保存时用了 DataParallel，key 会有 'module.' 前缀，需要去掉
    if list(model_state_dict.keys())[0].startswith('module.'):
        model_state_dict = {k[7:]: v for k, v in model_state_dict.items()}
        
    model.load_state_dict(model_state_dict)
    
    # 开始可视化
    visualize_trajectory(model, opts, save_name="dual_model_process.gif")