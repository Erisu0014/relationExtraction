#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/2/15 14:17
@Author  : Erisu-
@contact: guoyu01988@163.com
@File    : Candidate.py
@Software: PyCharm
@Desc:  候选词
'''


class Candidate:
    def __init__(self, distance, vector):
        self.distance = distance
        self.vector = vector
