#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/9 10:19
# @Author  : Erisu
# @contact : guoyu01988@163.com
# @File    : train_more.py
# @Software: PyCharm
# @Desc    : 迭代训练模型
import torch
# from dataUtils import dataHelper
# from TextCNN import TextCNN

import math
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from CNNmodel.TextCNN import TextCNN
from CNNmodel.dataUtils import *

# 上次重训练后得到的可信三元组数据信息

time_train = {}
# 当前可信度位于中间范围的向量
save_true_predict = {}
# 历史最好acc
best_acc = 0.606667


def train_unit(unit_datas, old_train_numpy, numpySavePath):
    global time_train
    global save_true_predict

    # 加载预训练参数
    args = {}

    read_numpy = numpy.load(numpySavePath)
    predict_sentences = read_numpy['predict_sentences']
    # 模型加载与参数初始化
    max_len = predict_sentences.shape[1]
    word_dim = predict_sentences.shape[-1]
    args['max_len'] = max_len
    args['word_dim'] = word_dim
    args['n_class'] = 5
    net = TextCNN(max_len=max_len, word_dim=word_dim, n_class=6)
    net.cuda()
    net = torch.load("bestmodel/cnnModel.pth")
    net.eval()
    # 训练批次大小
    predict_epoch = 300
    predict_len = predict_sentences.shape[0]
    # 对得到的句子向量进行预测
    last_logits = predict(predict_epoch=predict_epoch,
                          predict_sentences=predict_sentences, args=args, net=net)

    # 三元组根据可信度分类
    left_tensors, middle_tensors, right_tensors = unit_datas.reliabilityHelper(Pt=0.8, Ps=0.4,
                                                                               sentences_vector=predict_sentences,
                                                                               predict_tensors=last_logits)
    print(len(left_tensors), len(middle_tensors), len(right_tensors))
    # print(middle_tensors)
    if time_train:
        # 将上次的训练结果加入
        right_tensors.update(time_train)
        time_train = {}
    # 将位于两个可信度之间的数据加入
    save_true_predict.update(middle_tensors)
    middle_tensors = {}
    #
    list_train_input = []
    list_train_output = []
    # 遍历并返回即将训练的向量输入
    for key, value in right_tensors.items():
        list_train_input.append(value[0])
        list_train_output.append(value[-1])
    right_tensors = {}
    # 对可信度高的数据模型进行训练
    numpy_words_train = numpy.asarray(list_train_input, dtype=numpy.float)
    numpy_labels_train = numpy.asarray(list_train_output, dtype=numpy.float)
    # 训练批次大小
    train_epoch = 300
    train_acc, test_acc, epoch_size = train(old_train_numpy=old_train_numpy, train_epoch=train_epoch,
                                            numpy_words_train=numpy_words_train, numpy_labels_train=numpy_labels_train,
                                            args=args, net=net)
    # 预测备选知识库
    list_predict_input = []
    # 对字典排序并找值
    sorted_dict = sorted(save_true_predict.items(), key=lambda x: x[0])
    for item in sorted_dict:
        list_predict_input.append(item[-1][0])
    numpy_words_predict = numpy.asarray(list_predict_input, dtype=numpy.float)
    # 感觉这玩意一定不为0
    if numpy_words_predict.shape[0] != 0:
        predict_logits = predict(predict_epoch=predict_epoch,
                                 predict_sentences=numpy_words_predict, args=args, net=net)

        pre_relation = torch.max(predict_logits, 1)[1]
        for index, item in enumerate(sorted_dict):
            item[-1][1] = predict_logits[index]
            item[-1][-1] = pre_relation[index]
        # 三元组根据可信度分类
        left_tensors, middle_tensors, right_tensors = unit_datas.reliabilityHelper(Pt=0.8, Ps=0.4,
                                                                                   sentences_vector=numpy_words_predict,
                                                                                   predict_tensors=predict_logits)
        print(len(left_tensors), len(middle_tensors), len(right_tensors))
        # 用以下次训练
        time_train.update(right_tensors)
        # 将结果从排序字典写回dict
        temp_dict = {}
        for item in save_true_predict.items():
            temp_dict[item[0]] = item[1]
        save_true_predict = {}
        for key, value in temp_dict.items():
            if key in middle_tensors:
                save_true_predict[key] = value

    else:
        print("位于可信度区间的待抽取三元组有0个")
    return train_acc, test_acc, epoch_size, len(save_true_predict)


def predict(predict_epoch, predict_sentences, args, net):
    """

    :param predict_epoch: 每次预测的batch
    :param predict_sentences: 预测数据
    :param args: 词维度等信息
    :param net: 神经网络
    :return: 预测值
    """
    predict_len = predict_sentences.shape[0]
    last_logits = None
    for batch_index in range(0, math.ceil(predict_len / predict_epoch)):
        index_begin = predict_epoch * batch_index
        index_end = predict_epoch * (batch_index + 1)
        if index_end < predict_len:
            predict_input = predict_sentences[index_begin:index_end]
        elif index_begin == predict_len:
            # 恰好整除
            continue
        else:
            predict_input = predict_sentences[index_begin:]
        # numpy转torch
        torch_train_input = torch.from_numpy(
            predict_input.reshape(-1, 1, args['max_len'], args['word_dim'])).float().cuda()
        # print(torch_train_input.shape)
        logits = net(torch_train_input)
        torch_train_input = None
        soft_logits = F.softmax(logits, dim=1)
        # soft_logits = logits
        if last_logits is not None:
            last_logits = torch.cat((last_logits, soft_logits), dim=0)
        else:
            last_logits = soft_logits
    # print("?", last_logits.shape)
    return last_logits


def train(old_train_numpy, train_epoch, numpy_words_train, numpy_labels_train, args, net):
    """
    :param old_train_numpy:初始拿到的训练数据和测试数据路径，主要用途是对训练结果进行验证
    :param train_epoch: 每轮训练的数据
    :param numpy_words_train: 训练的句向量输入
    :param numpy_labels_train: 训练的label输入
    :param args: 词维度等信息
    :param net: 神经网络
    :return:
    """
    global best_acc
    # 读取数据
    read_numpy = numpy.load(old_train_numpy)
    words_test = read_numpy['words_test']
    labels_test = read_numpy['labels_test']
    n_class = read_numpy['class_num']
    args['word_dim'] = words_test.shape[-1]
    args['n_class'] = len(n_class)
    # 测试批次大小
    test_epoch = 50
    test_len = words_test.shape[0]
    # 保存训练过程中的准确率
    train_acc = []
    # 保存测试过程中的准确率
    test_acc = []
    # 模型训练
    # 设置超参数
    LR = 0.001
    EPOCH = 5
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    # 损失函数
    criteon = nn.CrossEntropyLoss()
    train_len = numpy_words_train.shape[0]
    print("train_len:", train_len)
    if train_len != 0:

        for epoch in range(EPOCH):
            accuracy = 0
            index = 1
            for batch_index in range(0, math.ceil(train_len / train_epoch)):

                index_begin = train_epoch * batch_index
                index_end = train_epoch * (batch_index + 1)
                if index_end < train_len:
                    train_input = numpy_words_train[index_begin:index_end]
                    train_output = numpy_labels_train[index_begin:index_end]
                elif index_begin == train_len:
                    # 恰好整除
                    continue
                else:
                    train_input = numpy_words_train[index_begin:]
                    train_output = numpy_labels_train[index_begin:]
                # numpy转torch
                torch_train_input = torch.from_numpy(
                    train_input.reshape(-1, 1, args['max_len'], args['word_dim'])).float().cuda()
                torch_train_output = torch.from_numpy(train_output).long().cuda()
                logits = net(torch_train_input)
                loss = criteon(logits, torch_train_output)
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
            # 测试
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
                torch_test_input = torch.from_numpy(
                    test_input.reshape(-1, 1, args['max_len'], args['word_dim'])).float().cuda()
                torch_test_output = torch.from_numpy(test_output).cuda()
                # net = net.cuda()
                logits = net(torch_test_input)
                # 得到关系预测的数值
                pre_tuple = torch.max(logits, 1)[1].cuda()
                acc = (numpy.fromiter(pre_tuple.cpu(),
                                      dtype=numpy.int32) == torch_test_output.cpu().numpy()).sum() / \
                      torch_test_output.size()[0]
                accuracy += acc
                index = batch_index
            accuracy /= index + 1
            test_acc.append(accuracy)
            if best_acc < accuracy:
                best_acc = accuracy
                # 保存网络参数
                print("save this model")
                torch.save(net, "bestmodel/cnnModel.pth")
            print("当前迭代次数为:{},测试过程准确率为：:{}".format(epoch, accuracy))
    return train_acc, test_acc, range(1, len(train_acc) + 1)


def draw(train_acc, test_acc, epoch, stage, index):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.title('自优化模型结果分析')
    plt.plot(epoch, train_acc, 'g-', label='training accuracy')
    plt.plot(epoch, test_acc, 'r--', label='testing accuracy')
    plt.legend()  # 显示图例
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')
    plt.savefig("more_pic/model_result" + str(index) + stage + ".png")
    plt.close()
    # plt.show()


def draw_more(best_accs, epoch):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.title('样本准确率曲线')
    plt.plot(epoch, best_accs, color='green', label='epoch acc')
    plt.legend()  # 显示图例
    plt.xlabel('样本源编号')
    plt.ylabel('准确率')
    plt.savefig("model_result_final.png")
    # plt.show()


def process():
    global time_train
    global save_true_predict
    global best_acc
    best_accs = []
    bookPath = "datas/tempBook"
    numpySavePath = "datas/numpy"
    dictPath = "datas/all_word_dict.txt"
    word2vec_path = "datas/word2vec126.pkl"
    old_train_numpy = "datas/numpy.npz"
    # 最多保留的中间三元组数量
    MAX_SAVE = 40
    unit_datas = dataHelper(entity_path=dictPath, word2vec_path=word2vec_path)
    # unit_datas.bookHelper(bookPath, numpySavePath=numpySavePath)
    index = 0
    for fileName in os.listdir(numpySavePath):
        filePath = numpySavePath + "/" + fileName
        print("当前文件:", filePath)
        for _ in range(5):
            print("当前迭代过程:", _)
            train_acc, test_acc, epoch, count = train_unit(unit_datas, old_train_numpy, filePath)
            # 如果现在的中间范围三元组比较小，则退出循环并画图
            # if count < MAX_SAVE:
            #     draw(train_acc, test_acc, epoch, str(_))
            #     break
            print(train_acc, "/", test_acc)
            draw(train_acc, test_acc, epoch, str(_), index)
        # 循环结束，将仍处于中间范围的predict写入文件
        time_train = {}
        save_true_vectors = []
        for item in save_true_predict.items():
            save_true_vectors.append(item[-1][0])
        numpy.savez("datas/numpy_middle/" + fileName, predict_sentences=save_true_vectors)
        save_true_predict = {}
        best_accs.append(best_acc)
        index += 1
    draw_more(best_accs, index)


def predict_books():
    # 需要预测的书的目录
    bookPath = "datas/predictBook"
    # 预测产生的临时文件夹
    numpySavePath = "datas/tempNumpy"
    dictPath = "datas/all_word_dict.txt"
    word2vec_path = "datas/word2vec126.pkl"
    old_train_numpy = "datas/numpy.npz"
    # 最多保留的中间三元组数量
    MAX_SAVE = 40
    unit_datas = dataHelper(entity_path=dictPath, word2vec_path=word2vec_path)
    unit_datas.bookHelper(bookPath, numpySavePath=numpySavePath)
    # relationName对应label的反转
    labels_relation = {v: k for k, v in unit_datas.relations.items()}
    for fileName in os.listdir(numpySavePath):
        filePath = numpySavePath + "/" + fileName
        # 加载预训练参数
        args = {}
        read_numpy = numpy.load(filePath)
        predict_sentences = read_numpy['predict_sentences']
        tuple_words = read_numpy['tuple_words']
        tuple_sentences = read_numpy['tuple_sentences']
        # 模型加载与参数初始化
        max_len = predict_sentences.shape[1]
        word_dim = predict_sentences.shape[-1]
        args['max_len'] = max_len
        args['word_dim'] = word_dim
        args['n_class'] = 5
        net = TextCNN(max_len=max_len, word_dim=word_dim, n_class=6)
        net.cuda()
        net = torch.load("bestmodel/cnnModel.pth")
        net.eval()
        # 预测批次大小
        predict_epoch = 300
        predict_len = predict_sentences.shape[0]
        # 对得到的句子向量进行预测
        last_logits = predict(predict_epoch=predict_epoch,
                              predict_sentences=predict_sentences, args=args, net=net)
        # 最大概率对应index
        max_logits_index = torch.max(last_logits, 1)[1]
        max_logits = torch.max(last_logits, 1)[0]
        # 待写入的三元组
        tuples = []
        for index, tuple_word in enumerate(tuple_words):
            relation_name = labels_relation[max_logits_index[index].cpu().item()]
            if relation_name == '未知' or max_logits[index].cpu().item() < 0.8:
                continue
            # 拼接为 r,s,t,sentence
            str = tuple_word[0] + ',' + tuple_word[1] + ',' + \
                  relation_name + ',' + \
                  tuple_sentences[index]
            tuples.append(str)
        filePath = "datas/final_predict/predict.txt"
        write_file = open(filePath, encoding='utf8', mode='w')
        for tuple in tuples:
            write_file.write(tuple + '\n')


def predict_sentences(sentence_path):
    # 预测产生的临时文件夹
    numpySavePath = "datas/numpy_sentences"
    dictPath = "datas/all_word_dict.txt"
    word2vec_path = "datas/word2vec126.pkl"
    unit_datas = dataHelper(entity_path=dictPath, word2vec_path=word2vec_path)
    unit_datas.sentenceHelper(sentence_path, numpySavePath+"/numpy.npz")
    # relationName对应label的反转
    labels_relation = {v: k for k, v in unit_datas.relations.items()}
    for fileName in os.listdir(numpySavePath):
        filePath = numpySavePath + "/" + fileName
        # 加载预训练参数
        args = {}
        read_numpy = numpy.load(filePath)
        predict_sentences = read_numpy['words_predict']
        print(predict_sentences.shape)
        tuple_sentences = read_numpy['tuple_sentences']
        # 模型加载与参数初始化
        max_len = predict_sentences.shape[1]
        word_dim = predict_sentences.shape[-1]
        args['max_len'] = max_len
        args['word_dim'] = word_dim
        args['n_class'] = 5
        net = TextCNN(max_len=max_len, word_dim=word_dim, n_class=6)
        net.cuda()
        net = torch.load("bestmodel/cnnModel.pth")
        net.eval()
        # 预测批次大小
        predict_epoch = 300
        predict_len = predict_sentences.shape[0]
        # 对得到的句子向量进行预测
        last_logits = predict(predict_epoch=predict_epoch,
                              predict_sentences=predict_sentences, args=args, net=net)
        # 最大概率对应index
        max_logits_index = torch.max(last_logits, 1)[1]
        max_logits = torch.max(last_logits, 1)[0]
        # 待写入的三元组
        tuples = []
        for index, sentence in enumerate(tuple_sentences):
            part_triple = sentence.split(',', 2)
            relation_name = labels_relation[max_logits_index[index].cpu().item()]
            # 拼接为 r,s,t,sentence
            str = part_triple[0] + ',' + part_triple[1] + ',' + \
                  relation_name + ',' + \
                  part_triple[-1]
            tuples.append(str)
        filePath = "datas/final_predict/predict.txt"
        write_file = open(filePath, encoding='utf8', mode='w')
        for tuple in tuples:
            write_file.write(tuple + '\n')


# process()
# predict_books()
predict_sentences(sentence_path="datas/predictBook/predict.txt")
