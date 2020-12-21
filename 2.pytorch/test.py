#!/usr/bin/env python
# coding=utf-8

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt



# x = torch.normal(0, 1, size=(4,2))

mean = torch.zeros([100, 2])
std = torch.ones([100, 2])
x = torch.normal(mean, std)

print(x.numpy())



