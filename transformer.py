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

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_en = spacy.load('en_core_web_sm')
spacy_zh = spacy.load('zh_core_web_sm')

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_zh(text):
    return [tok.text for tok in spacy_zh.tokenizer(text)]


SRC = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)

TRG = Field(tokenize = tokenize_zh, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True, 
            batch_first = True)

train_data, valid_data, test_data = data.TabularDataset.splits(path = './data1/', 
                                                               format = 'csv', 
                                                               skip_header = True, 
                                                               train='train.csv', 
                                                               validation='valid.csv', 
                                                               test='test.csv', 
                                                               fields = [('SRC', SRC), ('TRG', TRG)])

print(f"Number of training examples: {len(train_data.examples)}")
print(f"Number of validation examples: {len(valid_data.examples)}")
print(f"Number of testing examples: {len(test_data.examples)}")

print(vars(train_data.examples[0]))

SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)

print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 256

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device,
    sort_key = lambda x: len(x.SRC),
    sort = True)


INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
PE_TYPE = 0 # fixed
TE_TYPE = 0 # 0:scale = sqrt(hid_dim); 1: scale = 1
HID_DIM = 256 # 512
ENC_LAYERS = 3 # 6
DEC_LAYERS = 3 # 6
ENC_HEADS = 8 # 4
DEC_HEADS = 8 # 4
PF = True # False 
ENC_PF_DIM = 1024 # 2048
DEC_PF_DIM = 1024 # 2048
PF_ACT = 'relu' # 'gelu' 'relu' ''
ENC_DROPOUT = 0.1 # 0 0.2
DEC_DROPOUT = 0.1 # 0 0.2

REVERSE_SRC = 0 # 1

INIT_DISTRIBUTION = 'uni' # 'nor'

OPTIMIZER = 'adam' # 'sgd'
LEARNING_RATE = 0.0005

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
    model = nn.DataParallel(model)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

f=open('./checkpoint/HID_DIM_{}_LAYERS_{}_HEADS_{}_PF_DIM_{}.txt'.format(HID_DIM,ENC_LAYERS,ENC_HEADS,ENC_PF_DIM),'w')
f.write('parameters_{}\n'.format(count_parameters(model)))

def initialize_weights_uniform(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def initialize_weights_normal(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_normal_(m.weight.data)

if INIT_DISTRIBUTION == 'uni':
    model.apply(initialize_weights_uniform)
    print('!!!')

if INIT_DISTRIBUTION == 'nor':
    model.apply(initialize_weights_normal)
    print('!!!')


if OPTIMIZER == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
if OPTIMIZER == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum=0.9)


criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 30
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = transformer.train(model, train_iterator, optimizer, criterion, CLIP, epoch)
    valid_loss = transformer.evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './checkpoint/HID_DIM_{}_LAYERS_{}_HEADS_{}_PF_DIM_{}.pt'.format(HID_DIM,ENC_LAYERS,ENC_HEADS,ENC_PF_DIM))
    
    f.write(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    f.write(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    f.write(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('./checkpoint/HID_DIM_{}_LAYERS_{}_HEADS_{}_PF_DIM_{}.pt'.format(HID_DIM,ENC_LAYERS,ENC_HEADS,ENC_PF_DIM)))
test_loss = transformer.evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

from torchtext.data.metrics import bleu_score


def trans_transformer(sentence, src_field, trg_field, model, max_len = 50):
    
    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('en_core_web_sm')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0)
    if torch.cuda.is_available():
        src_tensor = src_tensor.cuda()
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0)
        if torch.cuda.is_available():
            trg_tensor = trg_tensor.cuda()
        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:-1], attention


def calculate_bleu(data, src_field, trg_field, model, max_len = 50):
    
    trgs = []
    pred_trgs = []
    
    for datum in data:
        
        src = vars(datum)['SRC']
        trg = vars(datum)['TRG']
        
        pred_trg, _ = trans_transformer(src, src_field, trg_field, model, max_len)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs)

bleu_score = calculate_bleu(test_data, SRC, TRG, model)

f.write(f'BLEU score = {bleu_score*100:.2f}')
f.close()


