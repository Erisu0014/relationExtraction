#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/3/20 16:49
@Author  : Erisu-
@contact: guoyu01988@163.com
@File    : dataUtils.py
@Software: PyCharm
@Desc:  数据处理与词向量转换
'''
import os
import jieba

import pickle
import numpy
import itertools
import torch


class DataIndex:
    def __init__(self, word, begin, end):
        self.word = word
        self.begin = begin
        self.end = end

    def __gt__(self, other):
        """ 用以比较一个DataIndex是否包含另一个，如果包含则返回true
        :param other:
        :return:
        """
        if self.word.find(other.word) != -1:
            if self.begin <= other.begin and self.end >= other.end:
                return True
            else:
                return False
        else:
            return False


class dataHelper:
    def __init__(self, entity_path, word2vec_path):
        self.entity_path = entity_path
        jieba.load_userdict(entity_path)
        # 读取word2vec
        # '../wordsEmbedding/result/word2vec_xinchou.pkl'
        word2vec_file = open(word2vec_path, 'rb')
        self.all_word_vec = pickle.load(word2vec_file)
        self.MAX_LEN = 100
        self.WORD_DIM = 126  # 126+2=128
        relation_path = os.listdir("datas/relation")
        # 关系label
        self.relations = {}
        # relation_label = open("datas/relation_label", mode='w', encoding='utf-8')
        # 关系三元组
        self.all_triple = []
        # "../data/triple"
        # triple_file = open(triplePath, mode='w', encoding='utf-8')
        self.count = 0
        # relation_label.write('未知 0\n')
        # relations['未知'] = 0
        # 获取当前label的最大值
        self.relation_max = {}
        for file in relation_path:
            file_name = file.split(".")[0]
            self.relations[file_name] = self.count
            # relation_label.write(file_name + ' ' + str(self.count) + '\n')
            file_reader = open("datas/relation/" + file, "r", encoding='utf-8')
            index = 0
            for lines in file_reader.readlines():
                self.all_triple.append(lines.strip('\n'))
                # triple_file.write(lines.strip('\n') + '\n')
                index += 1
            self.relation_max[self.count] = index
            self.count += 1
            file_reader.close()
        # relation_label.close()

    def entity_initalize(self, entity_path):
        """ 领域词存储
        :param entity_path: 领域词文件
        :return: entity_words:  领域词
        """
        entity_file = open(entity_path, encoding="UTF8", mode="r")
        entity_words = []
        for word in entity_file.readlines():
            if word.strip('\n') != '':
                entity_words.append(word.strip('\n'))
        entity_words.sort(key=lambda x: (len(x), x), reverse=True)
        return entity_words

    def tripleHelper(self, numpySavePath):
        """
        :param triplePath: 三元组（含路径）的位置
        :param numpySavePath: CNN输入向量的的存放路径
        """
        # relation_path = os.listdir("datas/relation")
        # # 关系label
        # relations = {}
        # relation_label = open("datas/relation_label", mode='w', encoding='utf-8')
        # # 关系三元组
        # all_triple = []
        # # "../data/triple"
        # # triple_file = open(triplePath, mode='w', encoding='utf-8')
        # count = 0
        # # relation_label.write('未知 0\n')
        # # relations['未知'] = 0
        # # 获取当前label的最大值
        # relation_max = {}
        #
        # for file in relation_path:
        #     file_name = file.split(".")[0]
        #     relations[file_name] = count
        #     relation_label.write(file_name + ' ' + str(count) + '\n')
        #     file_reader = open("datas/relation/" + file, "r", encoding='utf-8')
        #     index = 0
        #     for lines in file_reader.readlines():
        #         all_triple.append(lines.strip('\n'))
        #         # triple_file.write(lines.strip('\n') + '\n')
        #         index += 1
        #     relation_max[count] = index
        #     count += 1
        #     file_reader.close()
        # relation_label.close()
        # triple_file.close()

        # 分词

        word_train_vector = []
        labels_train = []
        word_test_vector = []
        labels_test = []
        location = [self.MAX_LEN]
        relation_sum = [0] * self.count
        # print(len(all_triple))
        for triple in self.all_triple:
            triple_vec = []
            part_triple = triple.split(',', 3)
            begin, end = part_triple[0], part_triple[1]
            # 分词与拼接
            split_begin = part_triple[3].index(begin)
            split_end = part_triple[3].index(end)
            words = jieba.lcut(part_triple[3][:split_begin])
            words.append(begin)
            words.extend(jieba.lcut(part_triple[3][split_begin + len(begin):split_end]))
            words.append(end)
            words.extend(jieba.lcut(part_triple[3][split_end + len(end):]))
            # print(words)
            index0, index1 = words.index(begin), words.index(end)
            # 位置向量
            if len(words) > self.MAX_LEN:
                print(words)
            location0 = numpy.arange(start=-index0, stop=self.MAX_LEN - index0, dtype=numpy.int32).tolist()
            location1 = numpy.arange(start=-index1, stop=self.MAX_LEN - index1, dtype=numpy.int32).tolist()
            # location0 = numpy.arange(start=-index0, stop=len(words) - index0, dtype=numpy.int32).tolist()
            # location1 = numpy.arange(start=-index1, stop=len(words) - index1, dtype=numpy.int32).tolist()
            for index, word in enumerate(words):
                temp_list = [location0[index], location1[index]]
                # temp_list = []
                # 加入位置向量
                try:
                    word_location = numpy.append(self.all_word_vec[word], temp_list, axis=0)
                # 可能没有得到该词的词向量
                except:
                    random_numpy = numpy.random.rand(self.WORD_DIM)
                    word_location = numpy.append(random_numpy, temp_list, axis=0)
                triple_vec.append(word_location)
            # 插入为定长
            value = self.MAX_LEN - len(triple_vec)
            if value > 0:
                for i in range(0, value):
                    ndarray = numpy.zeros(shape=self.WORD_DIM + 2)
                    # ndarray = numpy.zeros(shape=self.WORD_DIM)
                    triple_vec.append(ndarray)
            # 训练集
            if relation_sum[self.relations[part_triple[2]]] < self.relation_max[self.relations[part_triple[2]]] * 0.85:
                # words_list
                word_train_vector.append(triple_vec)
                labels_train.append(self.relations[part_triple[2]])
                # relation_num增加
                relation_sum[self.relations[part_triple[2]]] += 1
            # 测试集
            else:
                # words_list
                word_test_vector.append(triple_vec)
                labels_test.append(self.relations[part_triple[2]])

        print(type(word_train_vector))

        numpy_words_train = numpy.array(word_train_vector, dtype=numpy.float)
        numpy_labels_train = numpy.array(labels_train, dtype=numpy.float)
        numpy_class_num = numpy.arange(start=0, stop=self.count)
        numpy_words_test = numpy.array(word_test_vector, dtype=numpy.float)
        numpy_labels_test = numpy.array(labels_test, dtype=numpy.float)
        # 模型保存  # numpy.npz
        numpy.savez(numpySavePath, labels_train=numpy_labels_train, words_train=numpy_words_train,
                    class_num=numpy_class_num, words_test=numpy_words_test, labels_test=numpy_labels_test)

    def bookHelper(self, bookPath, numpySavePath):
        """

        :param bookPath:
        :param numpySavePath:
        :return:
        """
        # 此处已排序完成
        entity_words = self.entity_initalize(self.entity_path)
        # 保存的词向量
        word_vector = []
        # 保存的词关系及句子

        count = 0
        for book in os.listdir(bookPath):
            tuple_words = []
            tuple_sentences = []
            read_file = open(bookPath + "/" + book, mode="r", encoding="UTF8")
            for line in read_file.readlines():
                line = line.strip('\n')
                if line != "":
                    words_index = []
                    # jieba分词拿到的词可能不是最佳词
                    # words = jieba.lcut(line)
                    # for word in words:
                    #     if word in entity_words and word not in words_list:
                    #         words_list.append(word)
                    for word in entity_words:
                        begin = line.find(word)
                        if begin != -1:
                            otherIndex = DataIndex(word, begin, begin + len(word))
                            temp_bool = False
                            # 进行words_list判断
                            for dataIndex in words_index:
                                if not dataIndex.__gt__(otherIndex):
                                    continue
                                else:
                                    temp_bool = True
                                    break
                            if not temp_bool:
                                words_index.append(otherIndex)
                    words_list = []
                    for dataIndex in words_index:
                        words_list.append(dataIndex.word)
                    if len(words_list) < 2:
                        continue
                    # 任意取两个全排列，[0]得到的是一维数组
                    temp_combines = []
                    temp_combines.append(list(itertools.permutations(words_list, 2)))
                    words_combines = temp_combines[0]
                    print(words_combines)
                    for words_combine in words_combines:
                        # 这才是分词
                        begin, end = words_combine[0], words_combine[1]
                        # 分词与拼接
                        # print(part_triple[3])
                        split_begin = line.index(begin)
                        split_end = line.index(end)
                        words = jieba.lcut(line[:split_begin])
                        words.append(begin)
                        words.extend(jieba.lcut(line[split_begin + len(begin):split_end]))
                        words.append(end)
                        words.extend(jieba.lcut(line[split_end + len(end):]))
                        if len(words) > self.MAX_LEN:
                            continue
                        # 上面为分段分词
                        triple_vec = []
                        index0, index1 = words.index(words_combine[0]), words.index(words_combine[1])
                        # 位置向量
                        location0 = numpy.arange(start=-index0, stop=self.MAX_LEN - index0, dtype=numpy.int32).tolist()
                        location1 = numpy.arange(start=-index1, stop=self.MAX_LEN - index1, dtype=numpy.int32).tolist()
                        for index, word in enumerate(words):
                            temp_list = [location0[index], location1[index]]
                            # 加入位置向量
                            try:
                                word_location = numpy.append(self.all_word_vec[word], temp_list, axis=0)
                            # 可能没有得到该词的词向量
                            except:
                                random_numpy = numpy.random.rand(self.WORD_DIM)
                                word_location = numpy.append(random_numpy, temp_list, axis=0)
                            triple_vec.append(word_location)
                        # 插入为定长
                        value = self.MAX_LEN - len(triple_vec)
                        if value > 0:
                            for i in range(0, value):
                                ndarray = numpy.zeros(shape=self.WORD_DIM + 2)
                                triple_vec.append(ndarray)
                        word_vector.append(triple_vec)
                        tuple_words.append([words_combine[0], words_combine[1]])
                        tuple_sentences.append(line.strip('\n'))
            predict_sentences = numpy.asarray(word_vector, dtype=numpy.float)
            numpy_words = numpy.array(tuple_words, dtype=numpy.str)
            numpy_sentences = numpy.array(tuple_sentences, dtype=numpy.str)
            # 模型保存  # numpy.npz
            numpy.savez(numpySavePath + "/numpy" + str(count) + ".npz",
                        predict_sentences=predict_sentences, tuple_words=numpy_words,
                        tuple_sentences=numpy_sentences)
            word_vector = []
            count += 1

    def reliabilityHelper(self, Pt, Ps, sentences_vector, predict_tensors):
        """
        三元组分类
        :param Ps:  可信阙值
        :param Pt:  保留阙值
        :param sentences_vector:句向量（可直接用以训练或预测等）
        :param predict_tensors:句子的预测向量
        :return:
        """
        predict_logits = torch.max(predict_tensors, 1)[0]
        predict_relations = torch.max(predict_tensors, 1)[1]
        middle_tensors = {}
        right_tensors = {}
        left_tensors = {}
        for index in range(predict_logits.shape[0]):
            # uuid_num = uuid.uuid4()
            # print(predict_logit)
            # print(predict_logits[index])
            uuid_num = index
            if predict_logits[index] >= Pt:
                # 映射关系 uuid-句向量—预测概率-预测关系
                right_tensors[uuid_num] = []
                right_tensors[uuid_num].append(sentences_vector[index])
                right_tensors[uuid_num].append(predict_logits[index])
                right_tensors[uuid_num].append(predict_relations[index])
            elif Ps <= predict_logits[index] < Pt:
                # 映射关系 uuid-句向量—预测概率-预测关系
                middle_tensors[uuid_num] = []
                middle_tensors[uuid_num].append(sentences_vector[index])
                middle_tensors[uuid_num].append(predict_logits[index])
                middle_tensors[uuid_num].append(predict_relations[index])
            else:
                # print(predict_tensors[index])
                # 映射关系 uuid-句向量—预测概率-预测关系
                left_tensors[uuid_num] = []
                left_tensors[uuid_num].append(sentences_vector[index])
                left_tensors[uuid_num].append(predict_logits[index])
                left_tensors[uuid_num].append(predict_relations[index])
        return left_tensors, middle_tensors, right_tensors

    def sentenceHelper(self, sentences_path, numpySavePath):
        """

        :param sentences_path:
        :param numpySavePath:
        :return:
        """
        word_train_vector = []
        predict_sentences = []
        file_reader = open(sentences_path, "r", encoding='utf-8')
        for lines in file_reader.readlines():
            predict_sentences.append(lines.strip('\n'))
        for triple in predict_sentences:
            triple_vec = []
            part_triple = triple.split(',', 2)
            begin, end = part_triple[0], part_triple[1]
            # 分词与拼接
            split_begin = part_triple[2].index(begin)
            split_end = part_triple[2].index(end)
            words = jieba.lcut(part_triple[2][:split_begin])
            words.append(begin)
            words.extend(jieba.lcut(part_triple[2][split_begin + len(begin):split_end]))
            words.append(end)
            words.extend(jieba.lcut(part_triple[2][split_end + len(end):]))
            # print(words)
            index0, index1 = words.index(begin), words.index(end)
            # 位置向量
            if len(words) > self.MAX_LEN:
                print(words)
            location0 = numpy.arange(start=-index0, stop=self.MAX_LEN - index0, dtype=numpy.int32).tolist()
            location1 = numpy.arange(start=-index1, stop=self.MAX_LEN - index1, dtype=numpy.int32).tolist()
            for index, word in enumerate(words):
                temp_list = [location0[index], location1[index]]
                # temp_list = []
                # 加入位置向量
                try:
                    word_location = numpy.append(self.all_word_vec[word], temp_list, axis=0)
                # 可能没有得到该词的词向量
                except:
                    random_numpy = numpy.random.rand(self.WORD_DIM)
                    word_location = numpy.append(random_numpy, temp_list, axis=0)
                triple_vec.append(word_location)
            # 插入为定长
            value = self.MAX_LEN - len(triple_vec)
            if value > 0:
                for i in range(0, value):
                    ndarray = numpy.zeros(shape=self.WORD_DIM + 2)
                    # ndarray = numpy.zeros(shape=self.WORD_DIM)
                    triple_vec.append(ndarray)
            word_train_vector.append(triple_vec)
        numpy_words_train = numpy.array(word_train_vector, dtype=numpy.float)
        print(numpy_words_train.shape)
        numpy_class_num = numpy.arange(start=0, stop=self.count)
        numpy_sentences = numpy.array(predict_sentences, dtype=numpy.str)
        # 模型保存  # numpy.npz
        numpy.savez(numpySavePath, words_predict=numpy_words_train,
                    class_num=numpy_class_num, tuple_sentences=numpy_sentences)
