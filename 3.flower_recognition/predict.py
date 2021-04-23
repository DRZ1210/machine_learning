import torch
from torchvision import transforms
from PIL import Image
from model import generate_vgg_model
import matplotlib.pyplot as plt
import json
import os
from os.path import join


def mytest_model(type_name, image_path, weights_path, json_path):
    # print('test function start')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # 加载json文件
    with open(json_path, 'r') as file:
        index_to_class = json.load(file)

    # transpose 输入的只能是PIL打开的Image或者是tensor
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    source_img = Image.open(image_path)
    # 对图片进行转换
    img = test_transform(source_img)
    # 为图片添加batch维度
    img = torch.unsqueeze(img, dim=0)

    # 实例化模型，并加载之前训练好的模型参数
    model = generate_vgg_model(vgg_name='VGG16', batch_normal=False, num_classes=5, init_weights=True, pre_train=False)
    model.to(device)
    model.load_state_dict(torch.load(weights_path))
    # print('load model down')

    # 获取模型输出
    model.eval()
    with torch.no_grad():
        model_output = model(img.to(device))  # shape: (1, 5)
        model_output = torch.squeeze(model_output)

        index = torch.argmax(model_output).item()
        # prob = torch.max(model_output).item()

        plt.title('real_class: %s,  pred_class: %s' % (type_name, index_to_class[str(index)]))
        # print('real_class: %s || pred_class: %s' % (type_name, index_to_class[str(index)]))
        plt.imshow(source_img)
        plt.show()


json_path = './index_to_class.json'
weights_path = '/home/dengruizhi/2.flower/1.code/save_model/VGG16_flower_10.pth'
source_path = '/home/dengruizhi/2.flower/flowers_dataset/1.flower_5_types/mytest'

for typename in os.listdir(source_path):
    type_path = join(source_path, typename)
    filenames = os.listdir(type_path)
    for filename in filenames:
        mytest_model(typename, join(type_path, filename), weights_path, json_path)

