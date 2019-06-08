#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/10 15:14
# @Author  : Erisu
# @contact : guoyu01988@163.com
# @File    : test.py
# @Software: PyCharm
# @Desc    : 数据预备等工作
from CNNmodel.dataUtils import *
import pickle
import numpy

datas = dataHelper(entity_path="datas/my_dict.txt", word2vec_path="datas/word2vec126.pkl")
# datas.tripleHelper("datas/numpy.npz")
file = open("datas/numpy.npz", 'rb')
wordEmbedding = numpy.load(file)
print("训练集数据shape:", wordEmbedding['words_train'].shape)
print("验证集数据shape:", wordEmbedding['words_test'].shape)
print(wordEmbedding['labels_train'].shape)
print(wordEmbedding['labels_test'].shape)
print(wordEmbedding['class_num'].shape)
