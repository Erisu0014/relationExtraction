#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/2/27 19:52
@Author  : Erisu-
@contact: guoyu01988@163.com
@File    : test2.py
@Software: PyCharm
@Desc:
'''

file = open("../deal_book/spider_result.txt", encoding='utf8', mode='r')
length = len(file.readlines()) // 30
file.close()
len_dict = {}
len_dict[0] = []
count = 0
index = 0
file = open("../deal_book/spider_result.txt", encoding='utf8', mode='r')
for line in file.readlines():
    if line.strip('\n') == "":
        continue
    if index <= length:
        len_dict[count].append(line)
        index += 1
    else:
        index = 0
        count += 1
        len_dict[count] = []
        len_dict[count].append(line)
for key, values in len_dict.items():
    temp_file = open("../CNNmodel/datas/tempBook/sp" + str(key) + ".txt", encoding='utf8', mode='w')
    for value in values:
        temp_file.write(value.strip('\n') + '\n')
file.close()
