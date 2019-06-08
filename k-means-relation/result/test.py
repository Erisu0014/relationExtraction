#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/10 11:02
# @Author  : Erisu
# @contact : guoyu01988@163.com
# @File    : test.py
# @Software: PyCharm
# @Desc    : 测试词向量
import pickle

f = open('word2vec.pkl', 'rb')
wordsEmbedding = pickle.load(f)
print(wordsEmbedding.wv["的"])
f.close()
