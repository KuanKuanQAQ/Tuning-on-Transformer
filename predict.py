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
start_time = time.time()


test_data = data.TabularDataset(
    path = './data1/test.csv', 
    format = 'csv', 
    skip_header = True, 
    fields = [('SRC', SRC), ('TRG', TRG)])                                                               
end_time = time.time()
epoch_mins, epoch_secs = epoch_time(start_time, end_time)
print(f'Time for Loading Dataset: {epoch_mins}m {epoch_secs}s')

#print(f"Number of testing examples: {len(test_data.examples)}")
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

'''
for i in range(1):
    #example = 'Last week, I went to the theater.'
    #example = 'I had a very good seat.'
    #example = 'The play was very interesting.'
    #example = 'I did not enjoy it.'
    #example = 'A young man and a young woman were sitting behind me.'
    example = 'Furthermore, vocational and advanced training and education are offered in several of the areas mentioned.'
    src = tokenize_en(example)
    
    print(f"src = {' '.join(src)}")

    #translation_lstm = trans_lstm(src[::-1], SRC, TRG, model_lstm, device)
    #translation_attention, attention = trans_attention(src, SRC, TRG, model_attention, device)
    #translation_convolutional, attention = trans_convolutional(src, SRC, TRG, model, device)
    translation_transformer, attention = trans_transformer(src, SRC, TRG, model_transformer, device)
    

    #print(f"prd_lstm = {''.join(translation_lstm)}")
    #print(f"prd_attention = {''.join(translation_attention)}")
    #print(f"prd_convolutional = {''.join(translation_convolutional)}")
    print(f"prd_transformer = {''.join(translation_transformer)}")
    #display_attention_transformer(src, translation_transformer, attention)
'''
from torchtext.data.metrics import bleu_score

def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):
    
    trgs = []
    pred_trgs = []
    
    for datum in data:
        
        src = vars(datum)['SRC']
        trg = vars(datum)['TRG']
        
        pred_trg, _ = trans_transformer(src, src_field, trg_field, model, device, max_len)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs)

bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)

print(f'BLEU score = {bleu_score*100:.2f}')

