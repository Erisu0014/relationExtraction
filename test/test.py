#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/3/20 14:15
@Author  : Erisu-
@contact: guoyu01988@163.com
@File    : test.py
@Software: PyCharm
@Desc:
'''
import jieba
import jieba.posseg as posseg
import numpy
import pickle
import torch
import argparse
import sys

# MAX_LEN = 35
# index = 5
# word2vec_file = open('../wordsEmbedding/result/word2vec_xinchou.pkl', 'rb')
# all_word_vec = pickle.load(word2vec_file)
# a = numpy.append(all_word_vec["薪酬"], [20], axis=0)
# print(a.shape)
# triple_vec = numpy.empty(shape=MAX_LEN, dtype=numpy.float)
# triple_vec = numpy.append(triple_vec, a, axis=0)
# lista = []
# lista.append(triple_vec)
# tri = numpy.asarray(lista, dtype=numpy.float)
# tri = numpy.expand_dims(tri, axis=0)
#
# print(tri.shape)
# print(tri[0][:])
# a = numpy.arange(100).reshape(10, -1)
# b = torch.from_numpy(a)
# pred = torch.max(b, 1)[1]
# # print(pred)
# a = (3, 4, 6, 5, 7, 8)
# b = torch.arange(3, 9)
# pred = numpy.fromiter(a, dtype=numpy.int)
# acc = (pred == b.numpy()).sum()/b.size(0)
# print(acc)
# print(sys.argv[-1])
# if sys.argv[-1] == "False":
#     print("false")
import jieba.posseg as posseg

# import itertools
#
# list1 = ['a', 'b', 'cc', 'd', 'e']
# list2 = []
# # for i in range(1, len(list1) + 1):
# i=2
# iter = itertools.permutations(list1, i)
# list2.append(list(iter))
# print(list2[0])
# print(list2[0][0])
# print(list2[0][0][0])

# import torch
#
# tensor = torch.rand(2, 3)
# print(tensor)
# print(torch.max(tensor, 1)[0], torch.max(tensor, 1)[1])
# print(torch.max(tensor, 1)[0][0])
# if torch.max(tensor, 1)[0][0] > 0.3:
#     print("zz")

# import uuid
#
# print(uuid.uuid4())
# print(uuid.uuid4())
# a={}
# a[uuid.uuid4()] = uuid.uuid4()

# D1 = {'name': 'Bob', 'age': [10, 20]}
# D3 = {'name': 'tom', 'height': '10px', 'born': '1997'}
# D1.update(D3)
# for key, value in D1.items():
#     print(key, ":", value)
#
# sorted_dict = sorted(D1.items(), key=lambda x: x[0])
# print(type(sorted_dict))
# print(sorted_dict[0][-1][-1])
# import torch
#
# tensor = torch.arange(10).view(-1, 5)
# for mm in tensor:
#     print(mm[-1])
# for _ in range(50):
#     print("hello world")
# a = torch.rand(4, 2, 28, 28)
# # 增加一个组
# print(a.unsqueeze(-1).shape)
# a = numpy.random.rand(10)
# print(a)
# print(a.shape)
# file = open("../CNNmodel/datas/word2vec.pkl",'rb')
# aa = pickle.load(file)
# print(aa['员工关系'])

# a = [1, 2, 3, 4, 5, 6, 8]
# print(type(numpy.array(a)))
# print(a[1:])
# import torch.nn as nn
#
# layer = nn.Conv2d(1, 64, kernel_size=(5, 128), stride=1, padding=(2, 0))
# print(layer.forward(torch.ones(10, 1, 20, 128)).size())
# import torch, numpy
#
# a = torch.ones(10, 3, 3)
# b = torch.ones(5, 3, 3)
# c = torch.ones(7, 3, 3)
# x = torch.cat((a, b, c), dim=0)
# s = x.view(x.shape[0], -1)
# print(s.shape)
import torch.nn as nn

file1 = open("正确.txt", encoding='UTF8', mode='r')
file2 = open("错误.txt", encoding='UTF8', mode='r')
a = {}
b = {}
a.setdefault('依据', [])
a.setdefault('决定', [])
a.setdefault('包含', [])
a.setdefault('正相关', [])
a.setdefault('负相关', [])
a.setdefault('因果', [])
b.setdefault('依据', [])
b.setdefault('决定', [])
b.setdefault('包含', [])
b.setdefault('正相关', [])
b.setdefault('负相关', [])
b.setdefault('因果', [])
for line in file1.readlines():
    if line.strip('\n') == '':
        continue
    line_parts = line.strip('\n').split(',')
    if line_parts[2] in a.keys():
        a[line_parts[2]].append(line.strip('\n'))
for line in file2.readlines():
    if line.strip('\n') == '':
        continue
    line_parts = line.strip('\n').split(',')
    if line_parts[2] in b.keys():
        b[line_parts[2]].append(line.strip('\n'))
file1.close()
file2.close()
for key, values in a.items():
    print("正确:" + key + ":", len(a[key]))
    print("错误:" + key + ":", len(b[key]))
    print("准确率:", len(a[key]) / (len(a[key]) + len(b[key])))
    file_temp = open("../temp/" + key + ".csv", encoding='utf8', mode='w')
    for value in values:
        file_temp.write(value + '\n')
    file_temp.close()
