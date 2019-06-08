#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/3/20 13:18
@Author  : Erisu-
@contact: guoyu01988@163.com
@File    : MLP.py
@Software: PyCharm
@Desc:
'''
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __int__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(784, 200),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)
        return x
