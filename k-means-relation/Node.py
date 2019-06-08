#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2018/11/26 16:02
@Author  : Erisu-
@contact: guoyu01988@163.com
@File    : Node.py
@Software: PyCharm
@Desc:  节点类
'''


class Node:

    def __init__(self, word, index):
        self.word = word
        self._i = 0
        self.order = index  # 序列号
        self.inputMap = dict()
        self.outputMap = dict()
        self.part = ""  # 词标

    def input_update(self, key, value):
        if key in self.inputMap.keys():
            temp_list = self.inputMap[key]
            temp_list.append(value)
            self.inputMap[key] = temp_list
        else:
            temp_list = list()
            temp_list.append(value)
            self.inputMap[key] = temp_list

    def output_update(self, key, value):
        if key in self.outputMap.keys():
            temp_list = self.outputMap[key]
            temp_list.append(value)
            self.outputMap[key] = temp_list
        else:
            temp_list = list()
            temp_list.append(value)
            self.outputMap[key] = temp_list

    def print_self(self):
        print("*" * 20)
        print(str(self.order) + ':' + self.word + '(' + self.part + ')')
        print('inputMap:')
        tempx = ""
        for key, value in self.inputMap.items():
            tempx = key + ':['
            for val in value:
                tempx = tempx + str(val.order) + ','
            tempx = tempx + ']'
        print(tempx)
        tempx = ""
        print('outputMap:')
        for key, value in self.outputMap.items():
            tempx = key + ':['
            for val in value:
                tempx = tempx + str(val.order) + ','
            tempx = tempx + ']'
        print(tempx)


