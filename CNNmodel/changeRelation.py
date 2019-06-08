#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 16:04
# @Author  : Erisu
# @contact : guoyu01988@163.com
# @File    : changeRelation.py
# @Software: PyCharm
# @Desc    : 得到未知关系
import os

file = open("datas/relation/未知.csv", encoding='utf8', mode='r')
file2 = open("datas/relation/未知.csv", encoding='utf8', mode='w')
for line in file.readlines():
    line_parts = line.strip('\n').split(',')
    line_parts[2] = '未知'
    strs=",".join(line_parts)
    file2.write(strs + '\n')
file.close()
file2.close()
