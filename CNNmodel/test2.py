#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/17 17:40
# @Author  : Erisu
# @contact : guoyu01988@163.com
# @File    : test2.py
# @Software: PyCharm
# @Desc    :


import os

tuples = []
for file_path in os.listdir("test_txt"):
    file = open("test_txt/" + file_path, mode="r", encoding="UTF-8")
    for line in file.readlines():
        line = line.strip('\n')
        if line.strip()=="":
            continue
        parts = line.split(',', 3)
        tuples.append(parts[0] + ',' + parts[1] + ',' + parts[-1])
    file.close()
file_final = open("test_txt/" + "test.txt", mode="w", encoding="UTF-8")
for tuple in tuples:
    file_final.write(tuple + '\n')
