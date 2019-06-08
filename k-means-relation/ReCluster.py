#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/2/16 1:12
@Author  : Erisu-
@contact: guoyu01988@163.com
@File    : ReCluster.py
@Software: PyCharm
@Desc:  存放k-means clusters及相邻k-means的inertia差值
'''


class ReCluster:

    def __init__(self, cluster, distance):
        self.cluster = cluster
        self.distance = distance
