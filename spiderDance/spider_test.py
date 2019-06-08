#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/15 13:24
# @Author  : Erisu
# @contact : guoyu01988@163.com
# @File    : spider_test.py
# @Software: PyCharm
# @Desc    :

from requests_html import HTMLSession


file = open("HRwords.txt", encoding='utf8')
file2 = open("spider_result.txt", mode='a', encoding='utf8')
entity_words = []
for line in file.readlines():
    entity_words.append(line.strip('\n'))

ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3514.0 Safari/537.36"

words = []
for word in entity_words:
    session = HTMLSession()
    url = "https://baike.baidu.com/item/" + word
    try:
        print(word)
        r = session.get(url, headers={'user-agent': ua})
        sel = 'body > div.body-wrapper > div.content-wrapper > div > div.main-content > div.para'
        finds = r.html.find(sel)
        if finds:
            for find in finds:
                # print(find.text)
                file2.write(find.text.strip('\n') + '\n')
        session.close()
        # time.sleep(5)
    except Exception as e:
        print(e)
        words.append(word)
    finally:
        session.close()
print("*" * 100)
while words:
    print(words)
    for word in words:
        session = HTMLSession()
        url = "https://baike.baidu.com/item/" + word
        try:
            r = session.get(url, headers={'user-agent': ua})
            sel = 'body > div.body-wrapper > div.content-wrapper > div > div.main-content > div.para'
            finds = r.html.find(sel)
            if finds:
                print(word)
                file2.write(finds.text.strip('\n') + '\n')
            session.close()
            words.remove(word)
            # time.sleep(5)
        finally:
            session.close()
file2.close()
