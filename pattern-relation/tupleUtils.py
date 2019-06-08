import os

os.system("python deal_data.py")
os.system("python rule.py")
os.system("python find_tuple.py")
os.system("python find_tuple_correlation.py")


def duplicate_remove():
    classify_path = os.listdir("three_zu/")
    for classify_file in classify_path:
        duplicate_list = list()
        f_r = open("three_zu/" + classify_file, "r", encoding='utf-8')
        for line in f_r.readlines():
            words = line.strip('\n').split(',')
            if len(words) == 4:
                bool = False
                for words_list in duplicate_list:
                    if words[0] == words_list[0] and words[1] == words_list[1] and words[2] == words_list[2]:
                        bool = True
                        break
                    else:
                        continue
                if not bool:
                    duplicate_list.append(words)
        f_r.close()
        f_w = open("three_zu/" + classify_file, "w", encoding='utf-8')
        for words_list in duplicate_list:
            words = ",".join(words_list)
            f_w.write(words.strip('\n') + '\n')


def save_to_one(base_path, out_path):
    write_file = open(out_path, encoding='utf8', mode='w')
    for i in os.listdir(base_path + "/three_zu"):
        file = open(base_path + "/three_zu/" + i, encoding="utf8", mode='r')
        for line in file.readlines():
            write_file.write(line.strip('\n') + '\n')


duplicate_remove()
save_to_one(".", "relation_test.txt")
