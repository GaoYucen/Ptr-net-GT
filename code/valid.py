#%%
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
import math

# 添加路径
import sys
sys.path.append('code')
from PointerNet import PointerNet
from Data_Generator import TSPDataset
from config import get_config

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

#%% 加载训练好的模型参数
print('Test Mode!')
model.load_state_dict(torch.load('param/param_' + str(params.nof_points) + '_' + str(params.nof_epoch) + '.pkl', weights_only=True))
print('load success')

#%% 定义计算路径长度的函数
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

#%%
model.eval()

# 读取测试数据
test_dataset = np.load('data/test.npy', allow_pickle=True)

test_dataloader = DataLoader(test_dataset,
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

#%% 定义CrossEntropyLoss()，并初始化losses
CCE = torch.nn.CrossEntropyLoss()
losses = []
batch_loss = []
iterator = tqdm(test_dataloader, unit='Batch')

length_list = []
length_opt_list = []
error_sum = 0
total_samples = 0

for i_batch, sample_batched in enumerate(iterator):
    test_batch = Variable(sample_batched['Points'])
    target_batch = Variable(sample_batched['Solutions'])

    if USE_CUDA:
        test_batch = test_batch.cuda()
        target_batch = target_batch.cuda()
    else:
        test_batch = test_batch.to(device)
        target_batch = target_batch.to(device)

    o, p = model(test_batch)

    solutions = np.array(p)
    points = np.array(test_batch.cpu())
    solutions_opt = np.array(target_batch.cpu())

    batch_error = 0

    for i in range(len(solutions)):
        length = get_length(points[i], solutions[i])
        length_opt = get_length(points[i], solutions_opt[i])
        length_list.append(length)
        length_opt_list.append(length_opt)
        error_opt = (length - length_opt) / length_opt * 100
        batch_error += error_opt

        # 输出当前路径长度和当前最优路径长度
        print(f"Batch {i_batch + 1}, Sample {i + 1}: Current Length = {length:.4f}, Current Optimal Length = {length_opt:.4f}")

    batch_error = batch_error / len(solutions)
    error_sum += batch_error * len(solutions)
    total_samples += len(solutions)

    average_error = error_sum / total_samples
    iterator.set_postfix(average_error=f"{average_error:.2f}%")

print(f"最终平均误差: {average_error:.2f}%")