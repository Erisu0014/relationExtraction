#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/2/27 21:28
@Author  : Erisu-
@contact: guoyu01988@163.com
@File    : create_pro.py
@Software: PyCharm
@Desc:
'''
import jieba

# from config import FLAGS

# test_file_name=FLAGS.test_data_path.split("/")[-1]
# f = open(FLAGS.data_path+test_file_name+"_unique_result.txt",encoding='utf-8')
# f_z = open(FLAGS.data_path+"z_"+test_file_name+"_unique_result.txt",encoding='utf-8')
f_n = open("real_data.txt", encoding='utf-8')
f_w = open("my_dict.txt", "w", encoding='utf-8')

m_list = set()
# for i in f:
#     i=i.strip()
#     m_list.add(i)

for j in f_n:
    m_list.add(j.strip())

# for j in f_z:
#     m_list.add(j.strip())

for j in m_list:
    fre = jieba.suggest_freq(j, tune=True)
    f_w.write(j + ' ' + str(fre) + " pro\n")

# f.close()
f_n.close()
f_w.close()
# f_z.close()
