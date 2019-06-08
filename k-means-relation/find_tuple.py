#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/1/25 11:48
@Author  : Erisu-
@contact: guoyu01988@163.com
@File    : find_tuple.py
@Software: PyCharm
@Desc:  三元组发现(基于词性)
'''
import re
import os
import sys as sys
import synonyms
from pyltp import SentenceSplitter
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import Parser
import warnings

warnings.filterwarnings("ignore")
LTP_DATA_DIR = 'D:/python/project/ltpModel/ltp_data_v3.4.0'  # ltp路径
sentences = set()  # 分离出的句子组
relations = dict()  # 关系组
entity_words = set()  # 用以后续词扩展的初始词
tuple = set()  # 最后得到的三元组结果
wrong_relation = ("工作")


def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)


def initialize(entity_file):
    """ 领域词存储

    :param entity_file: 领域词文件（包括扩展词）
    :return:
    """
    for word in entity_file.readlines():
        if word.strip() != '':
            entity_words.add(word.strip())
    str = "领域词存储完成"
    print("=" * 10, str, "=" * 10)
    return str


def deal_data(sentence):
    """ 对于明显无法确定关系的段落删除。

    :param sentence:
    :return:
    """
    # 1.去除括号中的内容（因为多半为补充声明没什么大用）
    sentence = re.sub("（.*）\.*|\(.*\)\.*", "", sentence)
    # 2.去除例如："图1-6　2006年的总体报酬模型"这样的句子
    # paragraph = re.sub("图\d+-\d+.+ ", "", paragraph)
    # 3.去除文献
    pattern = re.compile("\[.*\]")
    m = pattern.match(sentence)
    if m:
        if m.span(0)[0] == 0:
            sentence = ""
    # 4.判断一段话是否全是英文
    if re.sub(' ', "", sentence).encode('UTF-8').isalpha():
        sentence = ""
    # 5.判断是否为三元组候选句(遍历关系数组)
    # count = 0
    # for key in relations:
    #     if sentence.find(key) != -1:
    #         count = 1
    #         break
    # if count == 0:
    #     for values in relations.values():
    #         for relation in values:
    #             if sentence.find(relation) != -1:
    #                 count = 1
    #                 break
    # if count == 0:
    #     sentence = ""
    # 条件判断结束，sentence
    return sentence


def sentence_split(read_file):
    """ 对段落中的句子进行基于符号的划分

    :param read_file:   文件txt
    :return:    分好的句子存入到sequences了，所以只需要返回状态信息就好了
    """
    for paragraph in read_file.readlines():
        # 太短的段落（词？）没有分的必要了
        if paragraph == '' or len(paragraph) <= 4:
            continue
        sentence_splitter = SentenceSplitter.split(paragraph)
        for sequence in sentence_splitter:
            # 去除空行
            if sequence == '':
                continue
            # 二次分隔
            second_sentences = re.split('[，,]', sequence)
            for second_sentence in second_sentences:
                # 对于句子的筛选工作
                second_sentence = deal_data(second_sentence)
                if second_sentence == '' or len(second_sentence) <= 4:
                    continue
                sentences.add(second_sentence)
    str = "分句步骤已完成"
    print("=" * 10, str, "=" * 10)
    return str


def if_has_relation(words):
    """ 判断是否句子中有关系词的出现

    :param words:
    :return:
    """
    # 一句话抽象提取出的关系是否只有一个呢？下面的方法是按一个来做的
    relation_word = ""
    index = -1
    for word in words:
        for key in relations:
            if synonyms.compare(word, key) > 0.98 and key not in wrong_relation:
                index = 0
                relation_word = key
                break
        if index == -1:
            for values in relations.values():
                for relation in values:
                    if synonyms.compare(word, relation) > 0.98 and relation not in wrong_relation:
                        index = 0
                        relation_word = word
                        break
                if index != -1:
                    break
        if index != -1:
            break
    return index, relation_word


def has_same(s1, s2):
    """ 判断两个字符串是否有相同部分

    :param s1:
    :param s2:
    :return:
    """
    str_first = ""
    for temp in s1:
        str_first += temp
        if s2.find(temp) != -1:
            return True
    if str_first == s1:
        return False
    str_second = ""
    for temp in s2:
        str_second += temp
        if s1.find(temp) != -1:
            return True
    if str_second == s2:
        return False


def new_relation_find(words, sentence):
    """ 新关系发现

    :param words:
    :param sentence:
    :return:
    """
    # 存放三元组的字典
    tuple_dict = dict()
    index0 = -1
    index1 = -1
    bool = False
    for entity_word in entity_words:
        if sentence.find(entity_word) != -1:
            if tuple_dict:
                # 返回为true说明有重复部分
                if has_same(tuple_dict[index0], entity_word):
                    continue
                index1 = sentence.find(entity_word)
                tuple_dict[index1] = entity_word
                bool = True
                break
            else:
                index0 = sentence.find(entity_word)
                tuple_dict[index0] = entity_word
    if bool is False:
        return "", "", ""
    # 排序结果为list
    # tuple_dict = sorted(tuple_dict.items(), key=lambda d: d[0])
    words = "/".join(words).split("/")
    for key, value in tuple_dict.items():
        tuple_word = value
        words = init_words(tuple_word, words)
    # 对于已经重构的词进行词标注
    postagger = Postagger()  # 初始化实例
    pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
    postagger.load_with_lexicon(pos_model_path, 'data/postagger.txt')  # 加载模型
    postags = postagger.postag(words)  # 词性标注
    print('\t'.join(postags))
    postagger.release()  # 释放模型
    # 发现新关系
    relation_word = ""
    index_word = 0
    for index, postag in enumerate('\t'.join(postags).split('\t')):
        index_word += len(words[index])
        if index_word >= len(sentence):
            break
        if postag == 'v' and index_word - min(index0, index1) <= 2 and max(index0, index1) - index_word <= 2 \
                and not has_same(tuple_dict[index0], words[index]) and not has_same(tuple_dict[index1],
                                                                                    words[index]) \
                and words[index] not in wrong_relation:
            relation_word = words[index]
            break
    if relation_word == "":
        return "", "", ""
    return tuple_dict[min(index0, index1)], tuple_dict[max(index0, index1)], relation_word


def init_words(tuple_word, words):
    """ 词重构的过程

    :param tuple_word: 三元组的关键词
    :param words: 分词任务产生的词
    :return:
    """
    oldIndex = tuple_word
    new_words = list()
    for word in words:
        if tuple_word.find(word) == -1:
            new_words.append(word)
        elif tuple_word.find(word) == 0:
            tuple_word = tuple_word[len(word):]
            if len(tuple_word) == 0:
                new_words.append(oldIndex)
    return new_words


def tuple_get(words, sentence):
    """
    :param words: 待抽取的三元组关系词
    :param sentence: 待抽取的句子
    :return:
    """
    # 判断是否有关系词
    index, relation_word = if_has_relation(words)
    t1 = ''
    t2 = ''
    if index != -1:
        for entity_word in entity_words:
            if sentence.find(entity_word) != -1:
                if t1 != '':
                    if has_same(t1, entity_word) or has_same(relation_word, entity_word):
                        continue
                    t2 = entity_word
                    break
                else:
                    if has_same(relation_word, entity_word):
                        continue
                    t1 = entity_word
    # index=-1 根据动词规则视图发现新关系
    else:
        t1, t2, relation_word = new_relation_find(words, sentence)

    if t1 != '' and t2 != '':
        print("得到三元组关系为{t1},{t2},{relation},{sentence}".format(t1=t1, t2=t2, relation=relation_word, sentence=sentence))
        tuple.add(t1 + '\t' + t2 + '\t' + relation_word + '\t' + sentence)


def words_split():
    """ 对于句子进行分词

    :return:
    """
    segmentor = Segmentor()
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
    segmentor.load_with_lexicon(cws_model_path, 'data/all_word_dict.txt')
    for sequence in sentences:
        words = segmentor.segment(sequence)
        tuple_get(words, sequence)
    segmentor.release()


def create_relation(relation_file):
    for line in relation_file.readlines():
        words = line.strip("\n").split(" ")
        if words[1]:
            relations[words[0]] = words[1:]
    str = "关系字典已经创建"
    print("=" * 10, str, "=" * 10)
    return str


def write_to_file():
    file = open("result/result.txt", 'w', encoding='utf8')
    for singal_tuple in tuple:
        file.write(singal_tuple + '\n')


if __name__ == '__main__':
    entity_file = open('../data/all_word_dict.txt', 'r', encoding='utf8')
    # 初始化读词
    initialize(entity_file)
    # 读取关系词
    relation_file = open("../data/key_word.txt", "r", encoding="utf8")
    create_relation(relation_file)
    # 文件夹下读取待扩展的文件路径
    bookList = list()
    listdir("../book", bookList)
    for i in range(len(bookList)):
        read_file = open(bookList[i], "r", encoding='utf-8')
        # 分句
        sentence_split(read_file)
        # 分词
        words_split()
        # 写入文件
        write_to_file()
