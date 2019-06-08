#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/3/21 9:11
@Author  : Erisu-
@contact: guoyu01988@163.com
@File    : train.py
@Software: PyCharm
@Desc:  textCNN训练
'''
from CNNmodel.TextCNN import TextCNN as TextCNN
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
import matplotlib.pyplot as plt

# 加载预训练参数
args = {}
# 判断是否训练过了
Trained = False
triplePath = "../data/triple.csv"
numpySavePath = "numpy.npz"
if sys.argv[-1] == "True":
    Trained = True
if not Trained:
    triplePath = "../data/triple.csv"
    numpySavePath = "datas/numpy.npz"
else:
    # triplePath = "../data/triple_test.csv"
    # numpySavePath = "numpy_test.npz"
    triplePath = "../data/triple.csv"
    numpySavePath = "datas/numpy.npz"

# 读取数据
read_numpy = numpy.load(numpySavePath)
words_train = read_numpy['words_train']
labels_train = read_numpy['labels_train']
words_test = read_numpy['words_test']
labels_test = read_numpy['labels_test']
n_class = read_numpy['class_num']
# words.shaoe=(n,max_len,word_dim)
# labels.shape=(n)
args['max_len'] = words_train.shape[1]
args['word_dim'] = words_train.shape[-1]
args['n_class'] = len(n_class)

net = TextCNN(n_class=args['n_class'], max_len=args['max_len'], word_dim=args['word_dim'])
net.cuda()

# 设置超参数
LR = 0.005
EPOCH = 240
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
# 损失函数
criteon = nn.CrossEntropyLoss()
if Trained:
    net = torch.load("bestmodel/cnnModel.pth")
    net.eval()
# 训练批次大小
train_epoch = 150
train_len = words_train.shape[0]
# 测试批次大小
test_epoch = 50
test_len = words_test.shape[0]
# best model
best_acc = 0
# 保存训练过程中的准确率
train_acc = []
# 保存测试过程中的准确率
test_acc = []


def draw(train_acc, test_acc, epoch, LR):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.title('初始模型结果分析(' + str(best_acc) + ')')
    plt.plot(epoch, train_acc, color='green', label='training accuracy')
    plt.plot(epoch, test_acc, color='red', label='testing accuracy')
    plt.legend()  # 显示图例
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')
    plt.savefig("初始模型" + "(" + str(LR) + ")" + "训练结果.png")
    # plt.show()
    plt.close()


for epoch in range(1, EPOCH + 1):
    accuracy = 0
    index = 1
    for batch_index in range(0, math.ceil(train_len / train_epoch)):
        # index_begin = epoch[train_epoch] * batch_index
        # index_end = epoch[train_epoch] * (batch_index + 1)
        index_begin = train_epoch * batch_index
        index_end = train_epoch * (batch_index + 1)
        if index_end < train_len:
            train_input = words_train[index_begin:index_end]
            train_output = labels_train[index_begin:index_end]
        elif index_begin == train_len:
            # 恰好整除
            continue
        else:
            train_input = words_train[index_begin:]
            train_output = labels_train[index_begin:]
        # numpy转torch
        torch_train_input = torch.from_numpy(
            train_input.reshape(-1, 1, args['max_len'], args['word_dim'])).float().cuda()
        torch_train_output = torch.from_numpy(train_output).cuda()
        logits = net(torch_train_input)
        loss = criteon(logits, torch_train_output.long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 得到关系预测的数值 3 这里计算不知道对不对
        pre_tuple = torch.max(logits, 1)[1].cuda()
        true_pre = (numpy.fromiter(pre_tuple.cpu(), dtype=numpy.int32) == train_output).sum()
        acc = true_pre / torch_train_output.size()[0]
        accuracy += acc
        index = batch_index
    accuracy /= index + 1
    train_acc.append(accuracy)
    print("当前迭代次数为:{},训练过程准确率为：:{}".format(epoch, accuracy))

    # torch.save_old(net, "models/cnnModel_" + str(epoch) + ".pth")
    accuracy = 0
    index = 1
    for batch_index in range(0, math.ceil(test_len / test_epoch)):
        index_begin = test_epoch * batch_index
        index_end = test_epoch * (batch_index + 1)
        if index_end < test_len:
            test_input = words_test[index_begin:index_end]
            test_output = labels_test[index_begin:index_end]
        elif index_begin == test_len:
            # 恰好整除
            continue
        else:
            test_input = words_test[index_begin:]
            test_output = labels_test[index_begin:]
        # numpy转torch
        torch_test_input = torch.from_numpy(test_input.reshape(-1, 1, args['max_len'], args['word_dim'])).float().cuda()
        torch_test_output = torch.from_numpy(test_output).cuda()
        # net = net.cuda()
        logits = net(torch_test_input)
        soft_logits = F.softmax(logits, dim=1)
        # 得到关系预测的数值
        pre_tuple = torch.max(soft_logits, 1)[1].cuda()
        acc = (numpy.fromiter(pre_tuple.cpu(),
                              dtype=numpy.int32) == torch_test_output.cpu().numpy()).sum() / torch_test_output.size()[0]
        accuracy += acc
        index = batch_index
    accuracy /= index + 1
    test_acc.append(accuracy)
    if best_acc < accuracy:
        best_acc = accuracy
        # 保存网络参数
        print("save_old this model")
        torch.save(net, "bestmodel/cnnModel.pth")

    print("当前迭代次数为:{},测试过程准确率为：:{}".format(epoch, accuracy))

epochs = numpy.arange(1, EPOCH + 1)
print(best_acc)
draw(train_acc, test_acc, epochs, LR)
