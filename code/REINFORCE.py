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
from PointerNet_R import PointerNet
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
    model = PointerNet(params.embedding_size,
                       params.hiddens,
                       params.nof_lstms,
                       params.dropout,
                       params.bidir)

    #%% train mode
    print('Train mode!')
    #%% 读取training dataset
    train_dataset = np.load('data/test.npy', allow_pickle=True)

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

    #%% Training process
    for epoch in range(params.nof_epoch):
        batch_loss = []
        iterator = tqdm(dataloader, unit='Batch')
        for i_batch, sample_batched in enumerate(iterator):
            iterator.set_description('Epoch %i/%i' % (epoch + 1, params.nof_epoch))

            train_batch = Variable(sample_batched['Points'].float())
            target_batch = Variable(sample_batched['Solutions'])

            # 放置data到device
            if USE_CUDA:
                train_batch = train_batch.cuda()
                target_batch = target_batch.cuda()
            else:
                train_batch = train_batch.to(device)
                target_batch = target_batch.to(device)

            optimizer.zero_grad()

            batch_rewards = []
            batch_log_probs = []

            for i in range(train_batch.size(0)):  # 遍历每个样本
                point = train_batch[i].cpu().numpy()
                solution = []
                log_probs = []
                mask = np.zeros(params.nof_points)  # 用于记录已经访问过的节点
                current_node = 0  # 从节点 0 开始
                solution.append(current_node)
                mask[current_node] = 1

                for step in range(params.nof_points - 1):
                    input_tensor = torch.tensor(point).unsqueeze(0).float().to(device)
                    output = model(input_tensor, torch.tensor(mask).unsqueeze(0).float().to(device))
                    # 假设 output[0] 是动作概率分布的张量
                    action_probs_tensor = output[0]
                    probs = torch.softmax(action_probs_tensor[0, step], dim=-1)
                    # probs = torch.softmax(output[0, step], dim=-1)
                    action = torch.multinomial(probs, 1).item()  # 根据概率分布采样下一个节点
                    log_prob = torch.log(probs[action])
                    log_probs.append(log_prob)
                    solution.append(action)
                    mask[action] = 1

                # 计算奖励
                reward = -get_length(point, solution)  # 负的路径长度作为奖励
                batch_rewards.append(reward)
                batch_log_probs.append(torch.stack(log_probs).sum())

            # 计算损失
            # 将 batch_rewards 转换为 float32 类型的张量
            # batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
            batch_rewards = torch.tensor(batch_rewards).to(device)
            batch_log_probs = torch.stack(batch_log_probs)
            loss = -(batch_log_probs * batch_rewards).mean()

            batch_loss.append(loss.data.item())

            loss.backward()
            optimizer.step()

            # 更新进度条
            iterator.set_postfix(loss='{}'.format(loss.data.item()))

        # 更新进度条
        iterator.set_postfix(loss=sum(batch_loss) / len(batch_loss))

    #%% 存储模型
    torch.save(model.state_dict(), 'param/reinforce_param_' + str(params.nof_points) + '_' + str(params.nof_epoch) + '.pkl')
    print('save success')