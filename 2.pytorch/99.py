#!/usr/bin/env python
# coding=utf-8
import torch as t
x = t.rand(5,3)
y = t.rand(5,3)

print('no')
if t.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x+y)
    print('yes')

print('end')
