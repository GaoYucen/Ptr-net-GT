#%%
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np

# 添加路径
import sys
sys.path.append('code')
from RouteModel import RouteGenerator
from Data_Generator import TSPDataset
from config import get_config
from tqdm import tqdm


def get_length(point, solution):
    """
    计算路径长度
    """
    length = 0
    end = len(solution) - 1
    for i in range(end):
        length += np.linalg.norm(point[solution[i + 1]] - point[solution[i]])
    length += np.linalg.norm(point[solution[end]] - point[solution[0]])
    return length

if __name__ == '__main__':
    #%% 读取参数
    params, _ = get_config()

    #%% 根据是否使用gpu定义device
    if params.gpu and params.sys == 'win' and torch.cuda.is_available():
        USE_CUDA = True
        USE_MPS = False
        device = torch.device('cuda:0')
        print('Using GPU, %i devices.' % torch.cuda.device_count())
    elif params.gpu and params.sys == 'mac':
        USE_CUDA = False
        USE_MPS = True
        device = torch.device('mps')
        print('Using MPS')
    else:
        USE_CUDA = False
        USE_MPS = False
        device = torch.device('cpu')
        print('Using CPU')

    #%% 确定model参数
    model = RouteGenerator(
        node_dim=2,
        embed_dim=params.embedding_size,
        hidden_dim=params.hiddens,
        n_layers=params.nof_lstms,
        dropout=params.dropout
    )

    #%% train mode
    print('Train mode!')
    #%% 读取training dataset
    train_dataset = np.load('data/train.npy', allow_pickle=True)

    dataloader = DataLoader(train_dataset,
                            batch_size=params.batch_size,
                            shuffle=True,
                            num_workers=0)

    #%% 放置model到device
    if USE_CUDA:
        model.cuda()
        net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    else:
        model.to(device)

    # 定义优化器
    optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                  model.parameters()),
                           lr=params.lr)

    # 初始化最佳损失值为无穷大
    best_loss = float('inf')

    #%% Training process
    for epoch in range(params.nof_epoch):
        batch_loss = []
        iterator = tqdm(dataloader, unit='Batch')

        for i_batch, sample_batched in enumerate(iterator):
            iterator.set_description('Epoch %i/%i' % (epoch + 1, params.nof_epoch))

            train_batch = sample_batched['Points'].float().to(device)  # [batch_size, seq_len, 2]
            target_batch = sample_batched['Solutions'].to(device)

            optimizer.zero_grad()

            # 批量生成路径
            batch_size = train_batch.size(0)
            seq_len = params.nof_points

            # 初始化mask和log_probs
            mask = torch.zeros(batch_size, seq_len, device=device)
            log_probs = torch.zeros(batch_size, seq_len - 1, device=device)
            solutions = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            hidden = model.init_hidden(batch_size)  # 新增隐藏状态初始化

            # 起始节点设为0
            solutions[:, 0] = 0
            mask[torch.arange(batch_size), 0] = 1

            # 向量化路径生成
            for step in range(seq_len - 1):
                # 前向传播获取概率矩阵
                prob_matrix, hidden = model(
                    train_batch,
                    mask,
                    hidden=hidden
                )  # [B, L]

                # 采样动作
                actions = torch.multinomial(prob_matrix, 1).squeeze(-1)
                log_probs[:, step] = torch.log(torch.gather(prob_matrix, 1, actions.unsqueeze(-1)).squeeze())

                # 更新mask（最后一步不更新，保持路径闭合）
                if step < seq_len - 1:
                    mask = mask.scatter(1, actions.unsqueeze(1), 1)
                    solutions[:, step + 1] = actions

            # 计算奖励（向量化实现）
            # 计算闭合路径总长度
            points = train_batch
            sol_exp = solutions.unsqueeze(-1).expand(-1, -1, 2)
            seq_coords = torch.gather(points, 1, sol_exp)

            # 计算相邻节点距离
            diff = seq_coords[:, 1:] - seq_coords[:, :-1]
            dists = torch.norm(diff, dim=-1).sum(dim=1)

            # 闭合路径最后一步
            diff_last = seq_coords[:, 0] - seq_coords[:, -1]
            dists += torch.norm(diff_last, dim=-1)

            rewards = -dists  # 负路径长度作为奖励

            # 计算损失
            log_probs_sum = log_probs.sum(dim=1)
            loss = (log_probs_sum * rewards).mean()

            batch_loss.append(loss.data.item())

            loss.backward()
            optimizer.step()

            # 更新进度条
            iterator.set_postfix(loss='{}'.format(loss.data.item()))

            # 如果当前损失值小于最佳损失值，保存模型
            if loss.data.item() < best_loss:
                best_loss = loss.data.item()
                torch.save(model.state_dict(), 'param/best_loss_reinforce_param_' + str(params.nof_points) + '_' + str(
                    params.nof_epoch) + '.pkl')

        # 更新进度条
        iterator.set_postfix(loss=sum(batch_loss) / len(batch_loss))

        # torch.save(model.state_dict(),
        #            'param/reinforce_param_' + str(params.nof_points) + '_' + str(params.nof_epoch) + '.pkl')

    #%% 存储模型
    torch.save(model.state_dict(), 'param/reinforce_param_' + str(params.nof_points) + '_' + str(params.nof_epoch) + '.pkl')
    print('save success')