import os
import csv
import random
import pandas as pd

def read(from_en, from_zh):
    with open(from_en, mode = 'r') as en,\
         open(from_zh, mode = 'r') as zh:
        count_dict = {}
        pre = ""
        for _ in range(1000000):
            try:
                lines_en=en.readline().strip()
                lines_zh=zh.readline().strip()
            except Exception as e:
                print(e)
                break
            if judge_en(lines_en) and judge_zh(lines_zh) and lines_en != pre:
                words = lines_en.lower().split()
                for word in words:
                    count_dict[word] = count_dict[word] + 1 if word in count_dict else 1
    
                pre = lines_en
    i=0
    for word in count_dict:
        if count_dict[word] >= 2:
            i+=1
    return i

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            pass
        else:
            return False
    return True

def judge_zh(zh):
    if(is_Chinese(zh[0:-1]) and zh[-1]=='。'):
        return True
    return False


def judge_en(en):
    if len(en) <= 3:
        return False
    for item in en[:-1]:
        if item.isalpha() or item == " ":
            pass
        else:
            return False
    if(en[0].isupper() and en[1:].islower() and en[-1]=='.'):
        return True
    return False

if __name__=="__main__":

    root_dir=os.path.abspath('.')
    data_frm1=os.path.join(root_dir,'data1','UNv1.0.en-zh.en.txt')
    data_frm2=os.path.join(root_dir,'data1','UNv1.0.en-zh.zh.txt')

    #print(judge_en("What was equitable for one country might not be so for another."))
    print(read(data_frm1,data_frm2))
