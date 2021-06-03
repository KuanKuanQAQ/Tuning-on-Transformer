import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext import data
from torchtext.data import Field, BucketIterator

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time
import os

from seq2seq import transformer

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


spacy_en = spacy.load('en_core_web_sm')
spacy_zh = spacy.load('zh_core_web_sm')


def tokenize_en(text):
    '''
    Tokenizes German text from a string into a list of strings (tokens) and reverses it
    '''
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_zh(text):
    '''
    Tokenizes English text from a string into a list of strings (tokens)
    '''
    return [tok.text for tok in spacy_zh.tokenizer(text)]


SRC = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

TRG = Field(tokenize = tokenize_zh, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

# 说明一下：stit文件夹里保存的是：英文/中文单词到向量和向量到英文/中文单词的映射表，一共四个，对应四个文件。
# 之所以要保存这些文件，是因为要保证训练过程和预测过程所采用的向量映射表是相同的，这样才能正确地把单词输入到模型，和正确地把模型输出的向量翻译成单词。
# 所以这是必不可少的资源文件^^！
SRC.build_vocab()
TRG.build_vocab()

SRC.vocab.itos = np.load('./stit/SRC_itos.npy')
SRC.vocab.stoi = np.load('./stit/SRC_stoi.npy',allow_pickle=True).item()
TRG.vocab.itos = np.load('./stit/TRG_itos.npy')
TRG.vocab.stoi = np.load('./stit/TRG_stoi.npy',allow_pickle=True).item()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


INPUT_DIM = 3391 # 这两行不用动
OUTPUT_DIM = 4166 # 这两行不用动

# 以下参数是需要修改的
# 根据用户在网页上的输入修改以下参数
PE_TYPE = 0 # fixed
HID_DIM = 256 # 512
ENC_LAYERS = 6 # 3
DEC_LAYERS = 6 # 3
ENC_HEADS = 8 # 4
DEC_HEADS = 8 # 4
PF = True # False 
PF_ACT = 'relu' # 'gelu' 'relu' ''
ENC_PF_DIM = 1024 # 2048
DEC_PF_DIM = 1024 # 2048
TE_TYPE = 0 # 0:scale = sqrt(hid_dim); 1: scale = 1
ENC_DROPOUT = 0.1 # 0 0.2
DEC_DROPOUT = 0.1 # 0 0.2

enc = transformer.Encoder(INPUT_DIM, 
                        PE_TYPE, 
                        TE_TYPE, 
                        HID_DIM, 
                        ENC_LAYERS, 
                        ENC_HEADS, 
                        PF,
                        ENC_PF_DIM,
                        PF_ACT, 
                        ENC_DROPOUT)

dec = transformer.Decoder(OUTPUT_DIM, 
                        PE_TYPE, 
                        TE_TYPE, 
                        HID_DIM, 
                        DEC_LAYERS, 
                        DEC_HEADS, 
                        PF, 
                        DEC_PF_DIM, 
                        PF_ACT,
                        DEC_DROPOUT)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = transformer.Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX)
if torch.cuda.is_available():
    model = transformer.Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX).cuda()
model = torch.nn.DataParallel(model)

# 根据用户在网页上的输入，load对应的模型。
model.load_state_dict(torch.load('./checkpoint/HID_DIM256_LAYERS6_ENC_HEADS8_PFTrue_PF_ACTrelu_PF_DIM1024_TE_TYPE0_DROPOUT0.1.pt', map_location=device), False)
model = model.module

def trans_transformer(sentence, src_field, trg_field, model, device, max_len = 50):
    model = model.eval()
    if isinstance(sentence, str):
        nlp = spacy.load('en_core_web_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:], attention


# 注意此处的参数：
# 如果模型heads数为8，参数不变；
# 如果模型heads数为4，rows=4，cols=1。
def display_attention_transformer(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
    assert n_rows * n_cols == n_heads
    myfont = matplotlib.font_manager.FontProperties(fname='./SimHei.ttf') # fname指定字体文件
    fig = plt.figure(figsize=(6,15))
    for i in range(n_heads):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()
        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], fontproperties=myfont,
                        rotation=45)
        ax.set_yticklabels(['']+translation, fontproperties=myfont)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        
    plt.show()
    plt.close()

#example = 'Last week, I went to the theater.'
#example = 'I had a very good seat.'
#example = 'The play was very interesting.'
example = 'I did not enjoy it.'
#example = 'A young man and a young woman were sitting behind me.'
src = tokenize_en(example)
print(f"src = {' '.join(src)}")
translation_transformer, attention = trans_transformer(src, SRC, TRG, model, device)
print(f"prd_transformer = {''.join(translation_transformer[:-1])}")
#display_attention_transformer(src, translation_transformer, attention)
