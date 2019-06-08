# _*_ encoding:utf-8 _*_

import jieba

jieba.load_userdict("../data/my_dict.txt")
import re
import jieba.posseg as pseg
import operator
import os

# fa_zhi = 0.35
f_r = open("key_word.txt", "r", encoding='utf-8')
key_word = {}
fa_dict = {}
guan_dict = {}
for i in f_r:
    i = i.strip().split()
    key_word[i[0]] = i[1:]
    if len(i[1:]) >= 5:
        fa_dict[i[0]] = 1
    else:
        fa_dict[i[0]] = 1
    guan_dict[i[0]] = []
# fa_dict["正相关"] = 1
# fa_dict["负相关"] = 1
fa_dict["包含"] = 1
f_r.close()


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


for i in os.listdir("./../deal_book/"):
    f = open("./../deal_book/" + i, "r", encoding='utf-8')
    new_sents = [m for m in f.readlines() if m.strip()]
    for se in new_sents:
        se = se.strip()
        if not se:
            continue
        sen = jieba.lcut(se)
        temp_v = {}
        for g in key_word:
            temp = 0
            for w in sen:
                if w in key_word[g]:
                    temp += 1
            temp_v[g] = temp
            # temp_v[g]/=len(key_word[g])
        sorted_t = sorted(temp_v.items(), key=operator.itemgetter(1), reverse=True)
        for jj in sorted_t:
            if jj[1] >= fa_dict[jj[0]]:
                guan_dict[jj[0]].append(se)
                break
    f.close()

for k in guan_dict:
    f_w = open("pp/" + k + ".txt", "w", encoding='utf-8')
    for j in guan_dict[k]:
        word_ci = pseg.cut(j.strip())
        words = []
        for word, flag in word_ci:
            words.append([word, flag])
        shu = 0
        for word, flag in words:
            if flag == "pro":
                shu += 1
        if shu > 1 and "？" not in j:
            f_w.write(j + "\n")
    f_w.close()
