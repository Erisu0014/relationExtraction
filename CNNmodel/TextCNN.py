#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/3/20 20:54
@Author  : Erisu-
@contact: guoyu01988@163.com
@File    : TextCNN.py
@Software: PyCharm
@Desc:
'''
import torch.nn as nn
# import torch
# import numpy


class TextCNN(nn.Module):
    def __init__(self, **kwargs):
        super(TextCNN, self).__init__()
        self.n_class = kwargs['n_class']
        # print('n-class=', self.n_class)
        self.max_len = kwargs['max_len']
        self.word_location_dim = kwargs['word_dim']
        # n_class = 10
        # max_len = 32
        # word_location_dim = 128
        # (1,max_len,word_location_dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=(5, self.word_location_dim), stride=1,
                      padding=(2, 0)),
            nn.ReLU(),
            nn.MaxPool2d(2)  #
        )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, self.word_location_dim), stride=1,
        #               padding=(2, 0)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)  #
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, self.word_location_dim), stride=1,
        #               padding=(2, 0)),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)  #
        # )
        self.out = nn.Sequential(
            nn.Linear(256 * self.max_len // 2, self.n_class)

        )

    def forward(self, x):
        x = x.view(x.size(0), 1, self.max_len, self.word_location_dim)
        # a = self.conv1(x)
        # b = self.conv2(x)
        # c = self.conv3(x)
        # input = torch.cat((a, b, c), dim=1)
        # input = input.view(input.shape[0], -1)
        a = self.conv1(x)
        input = a.view(a.shape[0], -1)
        output = self.out(input)
        # output = F.softmax(output, dim=1)
        return output


if __name__ == "__main__":
    TextCNN(n_class=5, max_len=32, word_dim=128)
