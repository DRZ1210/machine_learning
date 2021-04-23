import torch
import torch.nn as nn
from torch import optim

from train_valid_function import train
from model import generate_vgg_model
from dataloader import generate_data


# device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 实例化模型
model = generate_vgg_model('VGG16', batch_normal=False, num_classes=5, init_weights=True, pre_train=True)
print('model down successfully')

# 数据预处理
data_path = {
    'train_path': '/home/dengruizhi/2.flower/flowers_dataset/1.flower_5_types/train',
    'valid_path': '/home/dengruizhi/2.flower/flowers_dataset/1.flower_5_types/validation'
}
batch_size = {'train': 6, 'valid': 6}
train_loader, valid_loader = generate_data(data_path, batch_size)
print('data_loader down successfully')

# 训练参数设置
EPOCHS = 10
save_interval = 10
valid_interval = 40
learning_rate = 1e-3

# 损失函数 优化器
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 开始训练
print('start train')
train(model=model,
      train_loader=train_loader,
      valid_loader=valid_loader,
      epochs=EPOCHS,
      optimizer=optimizer,
      criterion=criterion,
      device=device,
      valid_interval=valid_interval,
      save_interval=save_interval)
