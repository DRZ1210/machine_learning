import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


torch.manual_seed(1)

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = torch.pow(x, 2) + 0.2 * torch.rand(x.size())
x, y = Variable(x, requires_grad=False), Variable(y, requires_grad=False)


def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.02)
    loss_func = torch.nn.MSELoss()

    for step in range(1000):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy())

    torch.save(net1, 'net.pkl')  # 保存 entire net
    torch.save(net1.state_dict(), 'net_params.pkl')  # 只保存 parameters


def restore_net():
    net2 = torch.load('net.pkl')

    prediction = net2(x)

    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy())


def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    net3.load_state_dict(torch.load('net_params.pkl'))
    prediction = net3(x)

    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy())
    plt.show()

# save net
save()
# restre entire net
restore_net()
# restore only the net parameters
restore_params()


