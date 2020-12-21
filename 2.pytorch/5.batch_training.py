import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,  # 提取的数据集
    batch_size=BATCH_SIZE,  # batch size
    shuffle=True,  # 每一次epoch之后数据是否打乱
    num_workers=2  # 加载数据的线程数
)

# 过一遍整个数据集就是一次epoch
# 一个数据集会分为多个batch size训练(训练集很大有时候不可以一次性加载进神经网络)

# li = ['tom', 'jerry', 'jack', 'rose']
# print(list(enumerate(li)))
# # 输出: [(0, 'tom'), (1, 'jerry'), (2, 'jack'), (3, 'rose')]
# # 所以enumerate() 函数的作用就是将传入可迭代对象,输出可迭代对象和对应的下标

for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        print('epoch: ', epoch, ' | step: ', step, ' | batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())

# 这里的BATCH_SIZE和总数据整好整除,如果不整除,如何选择数据呢? 比如总数为10,batch_size为5
# 先按照batch_size提取数据,最后一次若不足batch_size 就把剩下的都提取出来
