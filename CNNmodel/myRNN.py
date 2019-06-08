#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/10 11:35
# @Author  : Erisu
# @contact : guoyu01988@163.com
# @File    : myRNN.py
# @Software: PyCharm
# @Desc    :
import torch
import torch.nn as nn

rnn = nn.RNN(100, 10)
print(rnn)
x = torch.randn(20, 3, 100)
out, h = rnn(x, torch.zeros(1, 3, 10))
print(out.shape, h.shape)
# 3,20,10
# 1,20,10
