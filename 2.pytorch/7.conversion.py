import torch
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F


# Hyper params
BATCH_SIZE = 50
LR = 0.05
DOWNLOAD_MNIST = False
EPOCH = 1


# get training data(batch training) and test data
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
# print(train_data.data.shape)  # torch.Size([60000, 28, 28])
# print(train_data.targets.shape)  # torch.Size([60000])
# plt.imshow(train_data.data[0].numpy())
# plt.title('show image, label: %d' % train_data.targets[0])
# plt.show()
loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

test_data = torchvision.datasets.MNIST(root='./mnist', train=False)

test_x = Variable(torch.unsqueeze(test_data.data, dim=1), requires_grad=False).type(torch.FloatTensor)[:2000] / 255.0
test_y = test_data.targets[:2000]

# print(test_x.shape)  # torch.Size([2000, 1, 28, 28])
# plt.imshow(test_x[0][0].data.numpy())  # torch.Size([2000])
# plt.title('test example, label: %d' % test_y[0].item())
# plt.show()
# print(test_y.shape)
# print(test_y[0].item())  # 对于数据内容为一个标量的tensor而言，获取内容不采用.data[0]而采用.item()


# start build CNN

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Sequential(  # (1, 28, 28)
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2  # output_size = (input_size + 2 * padding - kernel_size) / stride + 1
                # 当stride = 1时，公式简化为： padding = (stride - 1) / 2
            ),  # (16, 28, 28)
            torch.nn.ReLU(),   # (16, 28, 28)
            torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 若没说stride，默认与kernel_size相同
            # (16, 14, 14)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 5, 1, 2),  # (32, 14, 14)
            torch.nn.ReLU(),  # (32, 14, 14)
            torch.nn.MaxPool2d(kernel_size=2)  # (32, 7, 7)
        )
        self.output = torch.nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # (batch, 32, 7, 7) -> (batch, 32 * 7 * 7)
        x = self.output(x)
        return x


cnn = CNN()
print(cnn)


optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()


# start train
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(loader):
        b_x = Variable(x)
        b_y = Variable(y)

        output = cnn(x)
        loss = loss_func(output, b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:  # 每训练50次，用测试集进行测试，查看模型训练程度和损失
            output_y = cnn(test_x)
            # step or print(output_y.shape)
            pred_y = torch.max(output, dim=1)[1]  # torch.max()会返回两个list，第0个是真实的最大值，第二个是最大值对应的index，我们要的是index
            accuracy = sum(pred_y == y) / y.size()[0]

            print('Epoch: ', epoch, ' | step: ', step,
                  ' | accuracy: ', accuracy.item(), ' | loss: ', loss.item())


test_output = cnn(test_x[:10])
pred_output = torch.max(test_output, 1)[1].data.numpy()
print('prediction labels: ', pred_output)
print('real labels: ', test_y[:10].numpy())

torch.save(cnn.state_dict(), './mnist_cnn_params.pkl')



