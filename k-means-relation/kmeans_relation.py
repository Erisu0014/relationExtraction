#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/2/13 18:45
@Author  : Erisu-
@contact: guoyu01988@163.com
@File    : kmeans_relation.py
@Software: PyCharm
@Desc:  通过kmeans做关系词聚类
'''
from sklearn.cluster import KMeans
from sklearn import linear_model
import pickle
from sklearn.externals import joblib
from Candidate import Candidate
import numpy as np
from matplotlib import pyplot as plt
from ReCluster import ReCluster
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 词到向量的映射表
word_vector = dict()
# 聚类准则的变化趋势
inertias = list()


def deal_word():
    # 得到去重的关系词汇
    relation_file = open("../wordsEmbedding/relation_words.txt", "r", encoding="utf8")
    nonredundancy_words = set()
    for line in relation_file:
        # 判断一个词是否含英文
        line = line.strip('\n')
        bool = False
        for ch in line:
            if '\u4e00' <= ch <= '\u9fff':
                continue
            else:
                bool = True
                break
        if line not in nonredundancy_words and not bool:
            nonredundancy_words.add(line)
    # 得到模型训练的词向量
    word2vec_file = open('../wordsEmbedding/result/word2vec_xinchou.pkl', 'rb')
    all_data = pickle.load(word2vec_file)
    # 将词向量用list暂时存储起来
    list_data = list()
    for word in nonredundancy_words:
        # try:
        #     list_data.append(all_data[word])
        #     word_vector[word] = np.reshape(list_data[-1], newshape=[-1, 128])
        # except:
        #     continue
        list_data.append(all_data[word])
        word_vector[word] = np.reshape(list_data[-1], newshape=[-1, 128])
    return list_data


def get_words(centroids, list_data, labels, k):
    """ 得到聚类中心的相关候选词(最多5个)

    :param centroids: 聚类中心
    :param list_data: not numpy
    :param labels: labels数据集
    :param k: 聚类簇数
    :return:
    """
    classify_words = list()
    # 用以存放候选词的的list
    for i in range(0, k):
        classify_word = list()
        classify_words.append(classify_word)
    for index, label in enumerate(labels):
        classify_words[label].append(list_data[index])
    # print(classify_words)
    # 候选词距离计算
    candidate_words = list()
    for classify_word in classify_words:
        temp_candidates = [None] * 5
        index = 0
        for item in classify_word:
            numpy_item = np.asarray(item).reshape(1, 128)
            numpy_centroid = np.asarray(centroids[index]).reshape(1, 128)
            temp_distance = np.linalg.norm(numpy_centroid - numpy_item)
            bool = False
            for distance_order, candidate in enumerate(temp_candidates):
                if not candidate:
                    temp_candidates[distance_order] = Candidate(vector=item, distance=temp_distance)
                    bool = True
                    break
            # 如果temp_candidates已经被填满
            if not bool:
                for distance_order, candidate in enumerate(temp_candidates):
                    if candidate.distance > temp_distance:
                        temp_candidates[distance_order] = Candidate(vector=item, distance=temp_distance)
                        index_distance = distance_order
                        break
        candidate_words.append(temp_candidates)
        index += 1
    # print(candidate_words)
    # 从word_vec映射中找到word
    output_words = list()
    for candidate_word_set in candidate_words:
        for candidate_word in candidate_word_set:
            for key, value in word_vector.items():
                if candidate_word:
                    if (candidate_word.vector == value).all():
                        print(key, end='\t')
                        output_words.append(key)
        output_words.append("/")
    # 以文件形式输出
    write_file = open("../data/output_relation" + str(time.time())[8:-1] + ".txt", encoding="utf8", mode='w')
    for output_word in output_words:
        if output_word != "/":
            write_file.write(output_word + '\t')
        else:
            write_file.write("\n")


def train(num_clusters):
    list_data = deal_word()
    data = np.asarray(list_data).reshape(-1, 128)
    # print(data.shape)
    km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40, init='k-means++', n_jobs=-1)
    km_cluster.fit(data)
    # result = km_cluster.fit_predict(data)
    # [centroid, label] = cluster.dbscan(data, eps=0.2, min_samples=10)
    # print("聚类中心：", km_cluster.cluster_centers_)
    # print("聚类标签;", km_cluster.labels_)
    # # inertias：是K-Means模型对象的属性，它作为没有真实分类结果标签下的非监督式评估指标。
    # # 表示样本到最近的聚类中心的距离总和。距离越小越好，越小表示样本在类间的分布越集中。
    # print("聚类准则", km_cluster.inertia_)
    return km_cluster


def train_more(num_clusters, loop):
    """

    :param num_clusters: 聚类簇数
    :param loop: 训练循环次数
    :return:
    """
    k_means_list = list()
    for i in range(0, loop):
        km_cluster = train(num_clusters + i)
        k_means_list.append(km_cluster)
        # bool = False
        # # 判断是否有空
        # for (index, cluster) in enumerate(k_means_list):
        #     if not cluster:
        #         k_means_list[index] = km_cluster
        #         bool = True
        #         break
        # # 如果k_means_list不为空则进行inertia比较
        # if not bool:
        #     for (index, cluster) in enumerate(k_means_list):
        #         if km_cluster.inertia_ < cluster.inertia_:
        #             k_means_list[index] = km_cluster
        #             break
        # inertias.append(km_cluster.inertia_)
    # print(k_means_list)
    for (index, km_cluster) in enumerate(k_means_list):
        joblib.dump(km_cluster, '../kmeans_model/kmeans_relation' + str(index) + '.pkl')


def find_best_model(loop):
    all_km_clusters = list()
    for index in range(0, loop):
        km_cluster = joblib.load('../kmeans_model/kmeans_relation' + str(index) + '.pkl')
        distance = 1
        if index != 0:
            old_inertia = all_km_clusters[index - 1].cluster.inertia_
            recluster = ReCluster(km_cluster, abs(float(km_cluster.inertia_) - old_inertia))
            all_km_clusters.append(recluster)
        else:
            recluster = ReCluster(km_cluster, distance)
            all_km_clusters.append(recluster)
    # 对于头距离重定义
    all_km_clusters[0].distance = all_km_clusters[1].distance
    k_means_list = [None] * 5
    for recluster in all_km_clusters:
        bool = False
        # 判断是否有空
        for (index, cluster) in enumerate(k_means_list):
            if not cluster:
                k_means_list[index] = recluster
                bool = True
                break
        # 如果k_means_list不为空则进行inertia_distance比较
        if not bool:
            for (index, cluster) in enumerate(k_means_list):
                if recluster.distance > cluster.distance:
                    k_means_list[index] = recluster
                    break

    # for (index, inertia) in enumerate(inertias):
    #     if index != 0:
    #         print(float(inertia) - float(inertias[index - 1]), end='\t')
    return k_means_list, all_km_clusters


def show_inertia(num_clusters, loop):
    all_km_clusters = list()
    inertias = list()
    for index in range(0, loop):
        km_cluster = joblib.load('../kmeans_model/kmeans_relation' + str(index) + '.pkl')
        distance = 1
        if index != 0:
            old_inertia = all_km_clusters[index - 1].cluster.inertia_
            recluster = ReCluster(km_cluster, abs(float(km_cluster.inertia_) - old_inertia))
            all_km_clusters.append(recluster)
        else:
            recluster = ReCluster(km_cluster, distance)
            all_km_clusters.append(recluster)
        inertias.append(km_cluster.inertia_)
        # 对于头距离重定义
    all_km_clusters[0].distance = all_km_clusters[1].distance
    # 横坐标
    x_np = np.arange(start=num_clusters, stop=num_clusters + loop)
    x = x_np.tolist()
    # print(x)
    # 聚类准则指数
    # 图1.inertia之间的距离绝对值
    distances = list()
    for recluster in all_km_clusters:
        distances.append(recluster.distance)
    print(distances)
    plt.subplot(3, 1, 1)
    plt.title('inertia之间的距离绝对值')
    plt.plot(x, distances, color="red", linewidth=1)

    print(inertias)
    # 图2.每个model的inertia
    plt.subplot(3, 1, 3)
    plt.title('每个model的inertia')
    plt.plot(x, inertias, color="green", linewidth=1)
    plt.show()


if __name__ == '__main__':
    num_clusters = 10
    loop = 60
    # train_more(num_clusters, loop)
    # # # 将聚类准则写入文件
    # # inertia_file = open("../data/k-means-inertia.txt", encoding='utf8', mode='w')
    # # for item in inertias:
    # #     inertia_file.write(str(item) + '\n')
    # # 找到最好的几个模型
    # km_clusters, all_km_clusters = find_best_model(loop)
    # # 对于找到的比较好的k-means模型进行再存储
    # for (index, km_cluster) in enumerate(km_clusters):
    #     joblib.dump(km_cluster.cluster, '../data/kmeans_relation' + str(index) + '.pkl')
    # # 得到少数的关系候选词
    # for km_cluster in km_clusters:
    #     cluster = km_cluster.cluster
    #     get_words(cluster.cluster_centers_, deal_word(), cluster.labels_, len(cluster.cluster_centers_))
    show_inertia(num_clusters, loop)
