# _*_ encoding:utf-8 _*_
import re
import os


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


for i in os.listdir("../book"):
    f = open("../book/"+i, "r", encoding='utf-8')
# for i in os.listdir("../testBook"):
#     f = open("../testBook/"+i, "r", encoding='utf-8')
    if ".txt" not in i:
        continue
    f_w = open("../deal_book/" + i, "w", encoding='utf-8')
    baizhi = ["●", "·", "一、", "二、", "三、", "四、", "五、", "六、", "七、", "八、", "九、",
              "十、", "十一、", "十二、", "十三、", "十四、", "十五、", "十六、", "十七、", "十八、",
              "十九、", "二十、", "二十一、", "二十二、", "二十三、", "二十四、", "二十五",
              "(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)", "(9)", "(10)", "(11)", "(12)",
              "(13)", "(14)", "(15)", "(16)", "(17)", "(19)", "(20)"]
    zhong_fu = ["。", "！", "!", "？", "?"]
    tt = [m for m in f.readlines() if m.strip()]
    paragraph = []
    for i, line in enumerate(tt):
        is_find = False
        for bz in baizhi:
            if bz in line:
                for zf in zhong_fu:
                    line = line.replace(zf, "；")
                is_find = True
                break
        if i < len(tt) - 2:
            jishu = 0
            for bz in baizhi:
                if bz in tt[i + 1]:
                    jishu += 1
                    break
            for bz in baizhi:
                if bz in tt[i + 2]:
                    jishu += 1
                    break
            if jishu == 2 or (jishu == 1 and is_find == True):
                for zf in zhong_fu:
                    line = line.replace(zf, "；")
                line = line.replace("\n", "")
        paragraph.append(line)
    paragraph = "".join(paragraph)

    sentences = cut_sent(paragraph)

    for se in sentences:
        # print(se)
        if "\n" in se:
            for bz in baizhi:
                if bz in se:
                    se = se.replace("\n", "")
                    break
        f_w.write(se + "\n")
    f_w.close()
    f.close()
