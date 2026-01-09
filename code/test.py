#%%
import numpy as np

train_dataset = np.load('data/train.npy', allow_pickle=True)
test_dataset = np.load('data/test.npy', allow_pickle=True)

#%% 前几条数据
print(train_dataset[:3])
