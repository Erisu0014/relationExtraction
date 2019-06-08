#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    : 2019/2/27 11:19
@Author  : Erisu-
@contact: guoyu01988@163.com
@File    : find_tuple.py
@Software: PyCharm
@Desc:  无监督的方法进行关系抽取
'''
import jieba.posseg as pseg
import synonyms
import os
import jieba
import re

jieba.load_userdict("../data/my_dict.txt")
# segmentor = Segmentor()
# LTP_DATA_DIR = 'D:/python/project/ltpModel/ltp_data_v3.4.0'  # ltp路径
# cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
# # my_dict中有词标和词频
# segmentor.load_with_lexicon(cws_model_path, 'my_dict.txt')
# postagger = Postagger()  # 初始化实例
# pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
# postagger.load(pos_model_path)  # 加载模型
expand_all_words = []


def findindex(org, x):
    result = []
    for k, v in enumerate(org):
        if v == x:
            result.append(k)
    return result


# 连接词
correlation_words = ["和", "●", "、", "以及", "还有", "；", "及", "或"]
# 忽略词
ignore_words = ['不是', '并非', '能否', '避免', '可能', '大概', '是否']
# 近义词的临近概率
near_probability = 0.9
# 标注数组
labelB = ['n', 'nz', 'nt', 'ni', 'a', 'b', 'pro', 'j', 'i', 'nd']
labelE = ['n', 'nz', 'nt', 'ni', 'pro', 'j', 'i', 'nd']
connect_head_end = ['nd', 'u']

# 关系模板
templet = {}
entity_words = []



def initialize(entity_file):
    """ 领域词存储

    :param entity_file: 领域词文件（包括扩展词）
    :return:
    """
    for word in entity_file.readlines():
        if word.strip() != '':
            entity_words.append(word.strip())
    entity_words.sort(key=lambda x: (len(x), x), reverse=True)
    str = "领域词存储完成"
    print("=" * 10, str, "=" * 10)
    return str


def sort_by_something(words_file):
    words = []
    for line in words_file.readlines():
        line = line.strip('\n')
        words.append(line)
    words.sort(key=lambda x: (len(x), x), reverse=True)
    f2 = open('sorted_words', 'w', encoding='utf8')
    for word in words:
        f2.write(word + '\n')
    return words


def expand_words(t_i, flags, words):
    """ 单词扩展

    :param t_i: 领域词index
    :param flags: 词性数组
    :param words: 词数组
    :return:
    """
    t_c = words[t_i]
    temp_str = ""
    begin = t_i
    end = t_i
    while begin > 0:
        if flags[begin - 1] in labelB:
            temp_str = words[begin - 1] + temp_str
            begin = begin - 1
        else:
            break
    t_c = temp_str + t_c
    temp_str = ""

    # 对于‘u’的处理
    if begin > 0:
        if flags[begin - 1] in connect_head_end:
            temp_str = words[begin - 1]
            begin = begin - 1
            while begin > 0:
                if flags[begin - 1] in labelB:
                    temp_str = words[begin - 1] + temp_str
                    begin = begin - 1
                else:
                    break
            # 说明不只放入了一个u，u前有修饰
            if flags[begin] not in connect_head_end:
                t_c = temp_str + t_c
            temp_str = ""

    while end < len(flags) - 1:
        if flags[end + 1] in labelE:
            temp_str = temp_str + words[end + 1]
            end = end + 1
        else:
            break
    t_c = t_c + temp_str
    temp_str = ""

    # 对于'u'的镜像处理
    if end < len(flags) - 1:
        if flags[end + 1] in connect_head_end:
            temp_str = words[end + 1]
            end = end + 1
            while end < len(flags) - 1:
                if flags[end + 1] in labelE:
                    temp_str = temp_str + words[end + 1]
                    end = end + 1
                else:
                    break
            # 说明不只放入了一个uj，uj前有修饰
            if flags[end] not in connect_head_end:
                t_c = t_c + temp_str
            temp_str = ""
    t_c = t_c.strip()

    return t_c


def is_expanded(words, t_i, short_part, flags):
    bool = False
    t_c_entity = ""
    t_c_expand = ""
    t_c = None
    for entity_word in entity_words:
        if short_part.find(entity_word) != -1:
            if entity_word.find(words[t_i]) != -1:
                t_c_entity = entity_word
                bool = True
                break
    for expand_word in expand_all_words:
        if short_part.find(expand_word) != -1:
            if expand_word.find(words[t_i]) != -1:
                t_c_expand = expand_word
                bool = True
                break
    if not bool:
        t_c = expand_words(t_i, flags, words)
    else:
        if len(t_c_expand) >= len(t_c_entity):
            return t_c_expand
        else:
            return t_c_entity
    return t_c


def templet_get():
    # 读取关系模板的内容
    gui = open("rule", "r", encoding='utf-8')
    for i in gui:
        i = i.strip().split()
        templet.setdefault(i[0], [])
        templet[i[0]].append(i[1:])
    return templet


def write_to_file(f_three):
    # 排序
    tuple_list = sorted(all_three)
    for nn in tuple_list:
        # 长度限制
        if len(nn[0]) >= 2 and len(nn[1]) >= 2 and len(nn[3]) < 50:
            f_three.write(
                nn[0].strip('\n') + "," + nn[1].strip('\n') + "," + nn[2].strip('\n') + "," + nn[3].strip('\n') + "\n")
    f_three.close()
    return "success"


def tuple_get(group, part_index, entity_index, pattern_part, file_name, sentence):
    # 存放的头实体
    heads = []
    # 存放的尾实体
    ends = []
    my_r = []
    if group:
        for t in group:
            my_r.append(re.split("|".join(correlation_words), t))
    for i, short_sentence in enumerate(my_r):
        # 循环的原因是因为my_r可能在append的时候加入的是一个list，比如 水和地球，那么就是my_r[?]={水，地球}

        for short_part in short_sentence:

            # word_flag = pseg.lcut(short_part)
            # word_vector = segmentor.segment(short_part)
            # flag_vector = postagger.postag(word_vector)  # 词性标注
            # words = '\t'.join(word_vector).split('\t')
            # flags = '\t'.join(flag_vector).split('\t')
            # # 重定义flag
            # for index, word in enumerate(words):
            #     if word in entity_words:
            #         flags[index] = 'pro'
            word_flags = pseg.lcut(short_part)
            words = []
            flags = []
            for word, flag in word_flags:
                words.append(word)
                flags.append(flag)
            t_c = ""
            # 如果只有一个领域词
            if flags.count("pro") == 1:
                t_i = flags.index("pro")
                t_c = expand_words(t_i, flags, words)
                # 判断是否已经有是当前领域词的扩展词
                t_c = is_expanded(words, t_i, short_part, flags)
                # # 不扩展词
                # t_c=words[t_i]

            # 如果没有领域词
            elif flags.count("pro") == 0:
                # t_i = -1
                # for index_d, d in enumerate(flags):
                #     if "n" in d and len(d) > 1 and d != "nr":
                #         # 因为没有领域词于是强行加了一个n当领域词
                #         t_i = index_d
                #         break
                # if t_c == "":
                #     for index_d, d in enumerate(flags):
                #         if "n" in d:
                #             t_i = index_d
                #             break
                # if t_c == "":
                #     for i_d, d in enumerate(flags):
                #         if d == "v":
                #             t_i = i_d
                #             break
                # # 看起来又像是词扩展
                # if t_i > 0:
                #     if flags[t_i - 1] == "n" or flags[t_i - 1] == "a":
                #         t_c += words[t_i - 1]
                # if t_i != -1:
                #     t_c += words[t_i]
                # if t_i < len(flags) - 1:
                #     if flags[t_i + 1] == "n":
                #         t_c += words[t_i + 1]
                continue
            else:
                if part_index[i] == 0:
                    # # 尾临近原则
                    # t_i = findindex(flags, "pro")[-1]
                    # 尾疏远原则
                    t_i = findindex(flags, "pro")[-1]
                elif part_index[i] == len(pattern_part) - 1:
                    # 头临近原则
                    t_i = findindex(flags, "pro")[0]
                else:
                    # 总之还是临近
                    f_n = findindex(flags, "pro")
                    if f_n[0] < len(flags) - f_n[-1] - 1:
                        t_i = f_n[0]
                    else:
                        t_i = f_n[-1]
                # 总之又是词扩展
                t_c = is_expanded(words, t_i, short_part, flags)
                # 不扩展词
                # t_c = words[t_i]
            # 判断是不是要倒序，也就是谁是头实体，谁是尾实体
            if entity_index[i] == "A":
                heads.append(t_c)
            else:
                ends.append(t_c)
    # 如果都有了，那是好事
    old_length = len(all_three)
    if heads and ends:
        for head in heads:
            for end in ends:
                if head and end:
                    if head == end:
                        continue
                    else:
                        all_three.add(tuple([head, end, file_name, sentence]))
    if len(all_three) > old_length:
        return "success"
    else:
        return "fail"


def change_to_regular(relation_part):
    """
    关系模板转正则表达式
    :param relation_part:
    :return:
    """
    index = 0
    temp_pattern = ""
    entity_dict = {}
    part_dict = []
    for k, c in enumerate(relation_part):
        if c == "A":
            entity_dict[index] = "A"
            index += 1
            temp_pattern += "(.*)"
            part_dict.append(k)
        elif c == "B":
            entity_dict[index] = "B"
            index += 1
            temp_pattern += "(.*)"
            part_dict.append(k)
        else:
            temp_pattern += c
    return temp_pattern, entity_dict, part_dict


def change_to_regular_next(relation_part):
    """
        关系模板转正则表达式(近义词部分)
        :param relation_part:
        :return:
        """
    index = 0
    temp_pattern = ""
    entity_dict = {}
    part_dict = list()
    for k, c in enumerate(relation_part):
        if c == "A":
            entity_dict[index] = "A"
            index += 1
            temp_pattern += "(.*)"
            part_dict.append(k)
        elif c == "B":
            entity_dict[index] = "B"
            index += 1
            temp_pattern += "(.*)"
            part_dict.append(k)
        else:
            # 因为上面的过程抽取的词可能是有近义词情况的，所以对其近义词进行适度抽取
            jin = synonyms.nearby(c)
            if jin[0]:
                temp = []
                for w, x in zip(jin[0], jin[1]):
                    if x > near_probability:
                        temp.append(w)
                if len(temp) > 1:
                    temp_pattern += "(" + "|".join(temp) + ")"
                else:
                    temp_pattern += c
            else:
                temp_pattern += c
    return temp_pattern, entity_dict, part_dict


if __name__ == '__main__':
    f_r = open("key_word.txt", "r", encoding='utf-8')
    # 存储关系词
    key_word = {}
    for i in f_r:
        i = i.strip().split()
        key_word[i[0]] = i[1:]
    templet_get()
    entity_file = open('../data/all_word_dict.txt', 'r', encoding='utf8')
    initialize(entity_file)
    expand_file = open('../data/expand_words.txt', 'r', encoding='utf8')
    expand_all_words = sort_by_something(expand_file)
    # 已经分类筛选出的关键句路径
    classify_path = os.listdir("pp/")
    for classify_file in classify_path:
        file_name = classify_file.split(".")[0]
        if file_name in templet:
            # 用以存放所有的三元组
            all_three = set()
            f_r = open("pp/" + classify_file, "r", encoding='utf-8')
            f_three = open("three_zu/" + file_name, "w", encoding='utf-8')
            dai_chou = []
            for sentence in f_r.readlines():
                # 第一种pattern发现的三元组
                # 如果存在非关系，则先不进行处理
                ignore = False
                for ignore_word in ignore_words:
                    if sentence.find(ignore_word) != -1:
                        ignore = True
                        break
                if ignore:
                    continue
                for small_sentence in sentence.strip('。').split("，"):
                    # 通过bool判断那第一种情况是否成立
                    bool = False
                    # 第一种情况，直接可以根据templet找到三元组
                    for relation_part in templet[file_name]:
                        # 将关系模板改成正则表达式
                        temp_pattern, entity_index, part_index = change_to_regular(relation_part)
                        r = re.search(temp_pattern, small_sentence)
                        # 如果未成功匹配就继续
                        try:
                            t_r = r.groups()
                            if t_r:
                                state = tuple_get(t_r, part_index, entity_index, relation_part, file_name,
                                                  small_sentence)
                                if state == "success":
                                    bool = bool | True
                                    break
                                else:
                                    bool = bool | False
                        except:
                            continue
                    # 第二种pattern发现的三元组
                    if not bool:
                        for relation_part in templet[file_name]:
                            # 将关系模板改成正则表达式
                            temp_pattern, entity_index, part_index = change_to_regular_next(relation_part)
                            r = re.search(temp_pattern, small_sentence)
                            try:
                                group = r.groups()
                                if group:
                                    state = tuple_get(group, part_index, entity_index, relation_part, file_name,
                                                      small_sentence)
                            except:
                                continue
            # 写入文件
            write_to_file(f_three)
    f_r.close()
    # postagger.release()
    # segmentor.release()
