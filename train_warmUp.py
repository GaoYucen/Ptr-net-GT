import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model_dual_no_context import set_decode_type
from utils.log_utils import log_values
from utils import move_to


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    std_cost = torch.std(cost) / math.sqrt(len(cost))
    
    print('Validation overall avg_cost: {} +- {}'.format(avg_cost, std_cost))
    
    return avg_cost.item(), std_cost.item()


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


# --- 新增功能 1: 实时日志保存 ---
def save_epoch_log(opts, epoch, train_avg_cost, train_avg_loss, val_avg_cost, std_cost, epoch_duration):
    """
    将训练和验证指标实时追加写入 txt 文件
    """
    log_file = os.path.join(opts.save_dir, 'log.txt')
    
    # 如果文件不存在，先写入表头
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("epoch\ttrain_cost\ttrain_loss\tval_cost\tval_std\tduration(s)\ttimestamp\n")
    
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 使用追加模式 'a' 写入数据
    with open(log_file, 'a') as f:
        f.write(f"{epoch}\t{train_avg_cost:.6f}\t{train_avg_loss:.6f}\t"
                f"{val_avg_cost:.6f}\t{std_cost:.6f}\t{epoch_duration:.2f}\t{timestamp}\n")


# --- 新增功能 3: 模型保存机制 ---
def save_checkpoint(model, optimizer, baseline, epoch, opts, is_best=False):
    """
    保存模型参数：
    1. 保存当前 epoch (epoch-X.pt)
    2. 如果是 best，额外保存 best-model.pt
    """
    # print(f'Saving model checkpoint for epoch {epoch}...')
    
    state = {
        'epoch': epoch,
        'model': get_inner_model(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'baseline': baseline.state_dict()
    }
    
    # 1. 保存当前 epoch
    path = os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
    torch.save(state, path)
    
    # 2. 保存最优模型
    if is_best:
        best_path = os.path.join(opts.save_dir, 'best-model.pt')
        torch.save(state, best_path)
        print(f"  >>> [Checkpoint] New best model saved to {best_path}")


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts, best_val_cost=math.inf):
    print(f"\nStart train epoch {epoch}, lr={optimizer.param_groups[0]['lr']:.6f} for run {opts.run_name}")
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    epoch_train_costs = []
    epoch_train_losses = []

    if not opts.no_tensorboard:
        tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # --- 新增功能 2: 显式的课程学习切换逻辑 ---
    # 获取参数（使用 getattr 防止报错）
    enable_curriculum = getattr(opts, 'enable_curriculum', False)
    curriculum_epochs = getattr(opts, 'curriculum_epochs', 0)
    curriculum_start_size = getattr(opts, 'curriculum_start_size', 10)
    target_graph_size = opts.graph_size

    # 逻辑判断
    if enable_curriculum and epoch < curriculum_epochs:
        current_graph_size = curriculum_start_size
        print(f"  >>> [Curriculum ACTIVE] Epoch {epoch}: Warming up on SMALL size {current_graph_size} (Target: {target_graph_size})")
    else:
        current_graph_size = target_graph_size
        if enable_curriculum and epoch == curriculum_epochs:
             print(f"  >>> [Curriculum SWITCH] Epoch {epoch}: SWITCHING to TARGET size {current_graph_size}!")
        elif enable_curriculum:
             print(f"  >>> [Curriculum DONE] Epoch {epoch}: Training on target size {current_graph_size}")
        else:
             print(f"  >>> [Standard Training] Epoch {epoch}: Training on size {current_graph_size}")

    # 重新生成训练集 (使用 current_graph_size)
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=current_graph_size, 
        num_samples=opts.epoch_size, 
        distribution=opts.data_distribution
    ))
    
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=0)

    # 模型进入训练模式
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        cost, loss = train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )
        epoch_train_costs.append(cost.mean().item())
        epoch_train_losses.append(loss.item())
        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # 验证 (验证集永远是目标大小，不受课程学习影响，所以初期 Val Cost 高是正常的)
    avg_val_cost, std_val_cost = validate(model, val_dataset, opts)
    
    # 计算训练均值
    avg_train_cost = sum(epoch_train_costs) / len(epoch_train_costs) if epoch_train_costs else 0
    avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses) if epoch_train_losses else 0

    # 1. 保存日志
    save_epoch_log(
        opts, epoch, 
        avg_train_cost, avg_train_loss, 
        avg_val_cost, std_val_cost, 
        epoch_duration
    )

    if not opts.no_tensorboard:
        tb_logger.log_value('val_avg_reward', avg_val_cost, step)

    # 2. 判断最优并保存模型
    is_best = avg_val_cost < best_val_cost
    if is_best:
        best_val_cost = avg_val_cost
    
    save_checkpoint(model, optimizer, baseline, epoch, opts, is_best=is_best)

    baseline.epoch_callback(model, epoch)
    lr_scheduler.step()

    # 返回更新后的 best_val_cost 给 run.py
    return best_val_cost


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # 计算 Cost 和 LogLikelihood
    cost, log_likelihood = model(x)

    # 计算 Baseline
    bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # 计算 Loss
    reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    loss = reinforce_loss + bl_loss

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    if step % int(opts.log_step) == 0:
        log_values(cost, grad_norms, epoch, batch_id, step,
                   log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)
    
    # 返回分离的数值供日志记录
    return cost.detach(), loss.detach()