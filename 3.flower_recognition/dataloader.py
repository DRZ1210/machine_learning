import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import matplotlib.pyplot as plt


def generate_data(data_path, batch_size):
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_data = ImageFolder(root=data_path['train_path'], transform=train_transform)
    valid_data = ImageFolder(root=data_path['valid_path'], transform=valid_transform)

    # 创建从编号到名字的json文件
    class_and_index = train_data.class_to_idx
    index_and_class = dict(zip(class_and_index.values(), class_and_index.keys()))
    with open('index_to_class.json', 'w') as file:
        file.write(json.dumps(index_and_class, indent=4))
    print('index_to_class json down')

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size['train'], shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size['valid'])

    return train_loader, valid_loader


# data_path = {
#     'train_path': '/home/dengruizhi/2.flower/flowers_dataset/1.flower_5_types/train',
#     'valid_path': '/home/dengruizhi/2.flower/flowers_dataset/1.flower_5_types/validation'
# }
#
# batch_size = {'train': 4, 'valid': 4}
#
# train_loader, valid_loader = generate_data(data_path, batch_size)
#
# for index, (images, labels) in enumerate(train_loader):
#     nums = images.shape[0]
#     for i in range(nums):
#         plt.title(i)
#         plt.imshow(images[i].permute(1, 2, 0))
#         plt.show()
#     break
