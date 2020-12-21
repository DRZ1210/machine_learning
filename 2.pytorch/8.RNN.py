import torch
import torchvision  # 下载mnist时用到
import torch.utils.data as Data  # batch train 用到
import torch.nn.functional as F  # 某些激活函数
from torch.autograd import Variable  # Variable
import matplotlib.pyplot as plt  # plot


"""
循环神经网络：RNN， (recurrent neural network) 
会利用之前的状态，在RNN的基础上，现在比较好的有：LSTM RNN 长短期记忆神经网络
梯度弥散和梯度爆炸，这就是RNN不能想起久远记忆的原因

LSTM 在语音识别、图片描述、还有自然语言处理方面 大量使用,LSTM还有一个变体GRU
LSTM  长期状态和短期状态

输入控制、输出控制、忘记控制
主线和分线
"""

# Hyper params
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28  # rnn time step / image height (输入多少次) 考虑了多少时间点的数据
INPUT_SIZE = 28  # rnn input size / image width （每次输入多少数据） 每一个时间点需要输入多少数据
LR = 0.01
DOWNLOAD_MNIST = False

# 下载mnist并将数据转换为tensor的形式
train_data = torchvision.datasets.MNIST(
    root='./mnist',
    download=DOWNLOAD_MNIST,
    # transform： 将PIL.image 和 numpy.array类型的数据转换为tensor的形式，并且正则化 normalize in the range[0.0, 1.0]
    transform=torchvision.transforms.ToTensor(),
    train=True,
)

# print(train_data.data.shape)
# print(train_data.targets.shape)

# 将训练集数据封装成loader的形式，便于之后批训练
train_loader = Data.DataLoader(
    shuffle=True,
    num_workers=2,
    batch_size=BATCH_SIZE,
    dataset=train_data
)

# 获得测试集
test_data = torchvision.datasets.MNIST(root='./mnist', train=False, transform=torchvision.transforms.ToTensor())
test_x = Variable(test_data.data, requires_grad=False).type(torch.FloatTensor)[:2000] / 255.0
test_y = test_data.targets[:2000].numpy()

# print(test_x.shape)
# print(test_y)


class RNN(torch.nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = torch.nn.LSTM(  # 长短时记忆网络
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            # [batch_size, time_step, input_size] -> True, [time_step, batch_size, input_size] -> False
        )
        self.out = torch.nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # x (batch, time_step, input_size)
        # hidden state:  分线程的state，子线程的state
        out = self.out(r_out[: -1, :])  # (batch, time step, input)
        # 取最后一个output


rnn = RNN()
print(rnn)


optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = torch.nn.CrossEntropyLoss()
# cross entropy 输出的pred不是on hot的形式，输出的就是预测的label，换句话说就是已经做过softmax()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = Variable(x.view(-1, 28, 28))
        b_y = Variable(y)
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            accuracy = sum(pred_y == test_y) / test_y.size
            print('Epoch: ', epoch, ' | train loss: %.4f', loss.data[0], ' | test accuracy: ', accuracy)


test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()

print(pred_y, 'prediction number')
print(test_y[:10], 'real number')











