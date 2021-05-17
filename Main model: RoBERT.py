# -*- coding: utf-8 -*-
"""
Created on Fri May  7 11:23:24 2021

@author: Ivette Bonestroo
"""

import pickle

path = 'training_data.pickle'
path2 = 'test_data.pickle'

#loading train dataset
with open(path, 'rb') as file:
    train = pickle.loads(file.read())

#loading test dataset
with open(path2, 'rb') as file:
    test = pickle.loads(file.read())

!pip install transformers
from transformers import DistilBertModel, DistilBertForSequenceClassification
from transformers import DistilBertTokenizerFast
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
from robert_functions import remover_transcripts, preprocessing, chunks, iter_chunks, apply_chunking, BERT_tokenizer, elapsed_time, accuracy, BERT_model, RNN_df
from robert_functions import performance_average_bert, padding_emb, Recurrent, train_epoch, evaluate, train_RNN, remover_empty_rows, random_search_param, hyperparametertuning
import pandas as pd
from data_functions import pickle_splitter, binarizer

transtr, labeltr = pickle_splitter(train)
transtest, labeltest = pickle_splitter(test)

print(f"length train: {len(transtr)}, length test: {len(transtest)}")
print('Removing incomplete transcripts:')
transtr, labeltr = (remover_transcripts(transtr, labeltr))
transtest, labeltest = (remover_transcripts(transtest, labeltest))
print(f"length train: {len(transtr)}, length test: {len(transtest)}")
print("---")
transtr = preprocessing(transtr)
transtest = preprocessing(transtest)

y_tr = binarizer(labeltr)
y_test = binarizer(labeltest)

train_df = pd.DataFrame(transtr, columns = ['data'])
train_df['labels'] = y_tr
test_df = pd.DataFrame(transtest, columns = ['data'])
test_df['labels'] = y_test

from sklearn.model_selection import train_test_split
trainval_df = train_df.copy()
train_df, val_df = train_test_split(train_df, test_size=0.25, stratify = train_df['labels'], random_state = 2021)
test_df.head()
train_df.head()

#print(len(train_df), len(val_df), len(test_df))

import torch
import random
import numpy as np

seed = 2021
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
results, highest_robert, highest_bert = hyperparametertuning(train_df, val_df, tokenizer, 150)

length_chunks =  list(highest_robert.values())[0]
percentage_overlap = list(highest_robert.values())[1]
bert_lr = list(highest_robert.values())[2]
bert_epochs	= list(highest_robert.values())[3]
batch_size_RNN = list(highest_robert.values())[4]
RNN_epochs =  list(highest_robert.values())[5]
pos_weight = list(highest_robert.values())[6]
RNN_type = list(highest_robert.values())[7]
RNN_lr = list(highest_robert.values())[8]
RNN_wd = list(highest_robert.values())[9]
RNN_layers = list(highest_robert.values())[10]
RNN_units = list(highest_robert.values())[11]
dropout = list(highest_robert.values())[12]

import random
import torch
import pandas as pd
import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertModel, DistilBertForSequenceClassification
print('Results for RoBERT')
df_new_trainval = apply_chunking(trainval_df, length_chunks, percentage_overlap)
df_new_test = apply_chunking(test_df, length_chunks, percentage_overlap)
print("Chunking done.")
# removing empty rows
df_new_trainval = remover_empty_rows(df_new_trainval)
df_new_test = remover_empty_rows(df_new_test)
print()
print(df_new_trainval.head())
print(df_new_test.head())
print()
# tokenizing and mapping to input ID's 
print("Tokenizing...")
trainval_inputIDs, trainval_masks = BERT_tokenizer(df_new_trainval, tokenizer, max_len = length_chunks)
test_inputIDs, test_masks = BERT_tokenizer(df_new_test, tokenizer, max_len = length_chunks)
print("Tokenizing done.")
print()
### BERT
print("Transforming into tensors...")
trainval_labels = df_new_trainval['labels'].tolist()
trainval_inputs = torch.LongTensor(trainval_inputIDs)
trainval_labels = torch.LongTensor(trainval_labels)
trainval_masks2 = torch.FloatTensor(trainval_masks)
#print(trainval_inputs.shape, trainval_labels.shape, trainval_masks2.shape)
trainval_data = TensorDataset(trainval_inputs, trainval_masks2, trainval_labels)
trainval_dataloader_noshuffle = DataLoader(trainval_data, batch_size=16, shuffle = False) 
trainval_dataloader = DataLoader(trainval_data, batch_size=16, shuffle = True) 

test_labels = df_new_test['labels'].tolist()
test_inputs = torch.LongTensor(test_inputIDs)
test_labels = torch.LongTensor(test_labels)
test_masks2 = torch.FloatTensor(test_masks)
test_data = TensorDataset(test_inputs, test_masks2, test_labels)
test_dataloader = DataLoader(test_data, batch_size=16)
print("Transforming done.")
print()

# getting embeddings from BERT
print("finetuning BERT and getting BERT embeddings...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1_bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', output_hidden_states=True).to(device)
model1_bert.resize_token_embeddings(len(tokenizer)) 
tr_emb, test_emb, model, test_logits_list, tr_logits_list = BERT_model(model1_bert, trainval_dataloader,trainval_dataloader_noshuffle, test_dataloader, epochs = bert_epochs, 
                                lr = bert_lr, device = device) 
print("Getting BERT embeddings done.")

test_auc_bert, test_acc_bert = performance_average_bert(test_logits_list, test_df['labels'].tolist(), df_new_test['index'].tolist())
tr_auc_bert, tr_acc_bert = performance_average_bert(tr_logits_list, trainval_df['labels'].tolist(), df_new_trainval['index'].tolist())
print('test auc BERT:', test_auc_bert)
print('train auc BERT:', tr_auc_bert)

print()
### RNN
print('Creating new dataframes with embeddings per transcript for RNN...')
trainval_df_RNN, tr_length_embs, tr_emb_maxlen = RNN_df(tr_emb, df_new_trainval['index'].tolist(), trainval_df, length_chunks) 
test_df_RNN, tr_length_embs, test_emb_maxlen, = RNN_df(test_emb, df_new_test['index'].tolist(), test_df, length_chunks)
print(trainval_df_RNN.head())
print(test_df_RNN.head())
print('Done.')
print()

trainval_df_RNN['padded_data'] = trainval_df_RNN['embedding'].apply(padding_emb, args=(tr_emb_maxlen, length_chunks))
trainval_inputs2 = trainval_df_RNN['padded_data'].to_list()
trainval_labels2 = trainval_df_RNN['labels'].to_list()

seq_list_tr = torch.tensor(trainval_df_RNN['len'].tolist())
trainval_inputs2 = torch.FloatTensor(trainval_inputs2)
trainval_labels2 = torch.FloatTensor(trainval_labels2)
trainval_data2 = TensorDataset(trainval_inputs2,trainval_labels2, seq_list_tr)
trainval_dataloader2 = DataLoader(trainval_data2, batch_size=int(batch_size_RNN))

test_df_RNN['padded_data'] = test_df_RNN['embedding'].apply(padding_emb, args=(test_emb_maxlen, length_chunks))
test_inputs2 = test_df_RNN['padded_data'].to_list()
test_labels2 = test_df_RNN['labels'].to_list()
seq_list_test = torch.tensor(test_df_RNN['len'].tolist()) #list of number of embeddings for each transcript
test_inputs2 = torch.FloatTensor(test_inputs2)
test_labels2 = torch.FloatTensor(test_labels2)
test_data2 = TensorDataset(test_inputs2, test_labels2, seq_list_test)
test_dataloader2 = DataLoader(test_data2, batch_size=len(test_inputs2))

input_dim = trainval_inputs2.shape[2]
print(" Training RNN...")
model2 = Recurrent(recurrent = RNN_type, input_dim=input_dim, hidden_dim = RNN_units, num_layers=RNN_layers, output_dim=1, dropout = dropout).to('cuda') #RNN_layers
optimizer = torch.optim.Adam(model2.parameters(), lr = RNN_lr, weight_decay= RNN_wd)
history, tr_auc, test_auc = train_RNN(model2, trainval_dataloader2 ,test_dataloader2, pos_weight, optimizer, epochs=RNN_epochs, device = device)
print("RNN done.")
print()
print('test auc RoBERT:', test_auc)
print('train auc RoBERT:', tr_auc)

length_chunks = list(highest_bert.values())[0]
percentage_overlap = list(highest_bert.values())[1]
bert_lr = list(highest_bert.values())[2]
bert_epochs	= list(highest_bert.values())[3]

print('Results for BERT mean')
df_new_trainval = apply_chunking(trainval_df, length_chunks, percentage_overlap)
df_new_test = apply_chunking(test_df, length_chunks, percentage_overlap)
print("Chunking done.")
# removing empty rows
df_new_trainval = remover_empty_rows(df_new_trainval)
df_new_test = remover_empty_rows(df_new_test)
print()
print(df_new_trainval.head())
print(df_new_test.head())
print()
# tokenizing and mapping to input ID's 
print("Tokenizing...")
trainval_inputIDs, trainval_masks = BERT_tokenizer(df_new_trainval, tokenizer, max_len = length_chunks)
test_inputIDs, test_masks = BERT_tokenizer(df_new_test, tokenizer, max_len = length_chunks)
print("Tokenizing done.")
print()
### BERT
print("Transforming into tensors...")
trainval_labels = df_new_trainval['labels'].tolist()
trainval_inputs = torch.LongTensor(trainval_inputIDs)
trainval_labels = torch.LongTensor(trainval_labels)
trainval_masks2 = torch.FloatTensor(trainval_masks)
#print(trainval_inputs.shape, trainval_labels.shape, trainval_masks2.shape)
trainval_data = TensorDataset(trainval_inputs, trainval_masks2, trainval_labels)
trainval_dataloader_noshuffle = DataLoader(trainval_data, batch_size=16, shuffle = False) 
trainval_dataloader = DataLoader(trainval_data, batch_size=16, shuffle = True) 

test_labels = df_new_test['labels'].tolist()
test_inputs = torch.LongTensor(test_inputIDs)
test_labels = torch.LongTensor(test_labels)
test_masks2 = torch.FloatTensor(test_masks)
test_data = TensorDataset(test_inputs, test_masks2, test_labels)
test_dataloader = DataLoader(test_data, batch_size=16)
print("Transforming done.")
print()

# getting embeddings from BERT
print("finetuning BERT and getting BERT embeddings...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1_bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', output_hidden_states=True).to(device)
model1_bert.resize_token_embeddings(len(tokenizer)) 
tr_emb, test_emb, model, test_logits_list, tr_logits_list = BERT_model(model1_bert, trainval_dataloader,trainval_dataloader_noshuffle, test_dataloader, epochs = bert_epochs, 
                                lr = bert_lr, device = device) 
print("Getting BERT embeddings done.")

test_auc_bert, test_acc_bert = performance_average_bert(test_logits_list, test_df['labels'].tolist(), df_new_test['index'].tolist())
tr_auc_bert, tr_acc_bert = performance_average_bert(tr_logits_list, trainval_df['labels'].tolist(), df_new_trainval['index'].tolist())
print('test auc BERT:', test_auc_bert)
print('train auc BERT:', tr_auc_bert)
