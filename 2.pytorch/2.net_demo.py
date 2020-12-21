#!/usr/bin/env python
# coding=utf-8

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = torch.pow(x, 2) + 0.2 * torch.rand(x.size())

# plt.title("created data")
# plt.scatter(x, y, color='red')
# plt.show()

x, y = Variable(x), Variable(y)


class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.output(x)
        return x


net = Net(1, 16, 1)
print(net)

print("net.parameters: ", net.parameters())
plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()


for step in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy())
        plt.text(0.5, 0, 'loss=%.4f' % loss.item())
        plt.pause(0.1)

plt.ioff()
plt.show()



