#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/15 14:08
# @Author  : Erisu
# @contact : guoyu01988@163.com
# @File    : pattern_chinese.py
# @Software: PyCharm
# @Desc    : 对中文的正则匹配及词汇查找
import re

def filter(string):
    if not re.search('([1-9])+', string):
        filtrate = re.compile(u'[^\u4E00-\u9FA5]')  # 非中文
        filtered_str = filtrate.sub(r' ', string)  # replace
        return filtered_str.strip('\n')


file = open("HRwordsCE.txt", encoding='utf8', mode='r')
entitys = []
for line in file.readlines():
    new_str = filter(line)
    # for batch in line.strip('\n'):
    #     if batch.isspace() or ord(batch) in range(97, 123) or ord(batch) in range(65, 91):
    #         continue
    #     else:
    #         new_str.append(batch)
    # new_str = "".join(new_str)
    if new_str and not new_str.isspace():
        entitys.append(new_str.strip(' '))
print(entitys)
file2 = open("HRwords.txt", encoding='utf8', mode='w')
for entity in entitys:
    file2.write(entity + '\n')
file2.close()


