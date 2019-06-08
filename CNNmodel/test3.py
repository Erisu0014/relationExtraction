#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/5/7 15:02
# @Author  : Erisu
# @contact : guoyu01988@163.com
# @File    : test3.py
# @Software: PyCharm
# @Desc    :
import numpy
import os

a = os.listdir("../book")
count = 0
for i in a:
    x = open("../book/" + i, encoding='utf8', mode='r')
    for emm in x.readlines():
        emm = emm.strip('\n')
        if emm != '':
            count += len(emm)
print(count)
