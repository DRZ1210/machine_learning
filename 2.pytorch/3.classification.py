import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


n_data = torch.ones(100, 2)

x0 = torch.normal(-2 * n_data, 1)
y0 = torch.ones(100)

x1 = torch.normal(2 * n_data, 1)
y1 = torch.zeros(100)

x = torch.cat([x0, x1], 0).type(torch.FloatTensor)
y = torch.cat([y0, y1], ).type(torch.LongTensor)

# plt.title('random points')
# plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=y)
# plt.show()

x, y = Variable(x), Variable(y)

# method1
class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x


net1 = Net(2, 10, 2)
print(net1)

# method 2

net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2)
)
print(net2)



#
# optimizer = torch.optim.SGD(net.parameters(), lr=0.008)
# loss_func = torch.nn.CrossEntropyLoss()
#
# plt.ion()
# plt.show()
#
# for step in range(200):
#     out = net(x)
#     loss = loss_func(out, y)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if step % 2 == 0:
#         plt.cla()
#         prediction = torch.max(F.softmax(out), 1)[1]
#         predict_y = prediction.data.numpy()
#         real_y = y.data.numpy()
#
#         plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=predict_y)
#         accuracy = sum(predict_y == real_y) / 200
#         plt.text(1.5, -4, 'accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
#         plt.pause(0.1)
#
# plt.ioff()
# plt.show()
#
