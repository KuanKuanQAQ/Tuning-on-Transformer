'''
unfold contractions
it's -> it is
i've -> i have

but still
tom'll -> tom'll
so i have to delete it

delete double sentences
"""how old are you?"" ""i am sixteen.""",“你几岁？”“我十六岁。”

'''
import csv
import spacy
import contractions


spacy_en = spacy.load('en_core_web_sm')
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def unfold(file_from, file_to):
    with open(file_from, 'r') as r, open(file_to, 'w', newline = '') as w:
        count_dict = {}
        reader = csv.reader(r)
        writer = csv.writer(w)
        lines = list(reader)
        for line in lines:
            line[0] = contractions.fix(line[0]).capitalize()
            if (line[0].find('.') != len(line[0]) - 1 and line[0].find('.') != -1) or\
               (line[0].find('?') != len(line[0]) - 1 and line[0].find('?') != -1):
                continue
            if line[0].find('\'') != -1 or line[0].find('-') != -1:
                continue
            words = tokenize_en(line[0].lower())
            for word in words:
                count_dict[word] = count_dict[word] + 1 if word in count_dict else 1
    
            writer.writerow([line[0], line[1]])
        i=0 #number of unique words
        for word in count_dict:
            if count_dict[word] >= 2:
                i+=1
        print(i)



unfold("./data1/test.csv","./data1/test_tmp.csv")
unfold("./data1/train.csv","./data1/train_tmp.csv")
unfold("./data1/valid.csv","./data1/valid_tmp.csv")
