"""
Created on Tue Apr  6 15:16:24 2021

@author: Ivette Bonestroo
"""

def remover_transcripts(trans, label):  
  """Removes transcripts that are shorther than 50 words"""
  x = []
  y = []
  for i in range(len(trans)):
    if len(trans[i].split()) >= 50:
      x.append(trans[i])
      y.append(label[i])
  return x, y

def preprocessing(transcripts):
    """Cleaning the transcripts by removing words between brackets and punctuation"""
    import re
    import numpy as np
    #all = []
    count = 0
    for transcript in transcripts:
   # remove words between [] and (), BERT is not trained on this
      new_stopwords = []
      z = re.findall("(\[.*?\])", transcript)
      x = re.findall("(\(.*?\))", transcript)
      for word in z:
          new_stopwords.append(word)
      for word in x:
          new_stopwords.append(word)
    # common stopwords that will also be removed
      new_stopwords.extend(['Music', 'music', 'MUSIC', 'Applause',  'APPLAUSE', 'applause']) 
          
      #print(new_stopwords)
      new_stopwords = list(set(new_stopwords))
      
      for word in new_stopwords:
          if len(word.split()) > 10: ## continue when sentence in stopwords is longer than 10 words
              ##sentences between brackets can be informative, a few words not really
              continue
          if word in transcript:
              transcript = transcript.replace(word, "") ##remove stopwords in stopwords list
      for word in "@#$%^&*{}♪<>'/'\?!.,[](;:)": ## removing non-alphanumerical characters
          if word in transcript:
              transcript = transcript.replace(word, "")
      transcript = transcript.replace("\n", " ") ## replacing inconsistencies with a space
      transcript = transcript.replace("xa0", " ")
           
      if count == 0:
        all = np.array([transcript])
        count += 1
      else:
        all = np.vstack((all, transcript))

    return all

import pandas as pd
import math
def chunks(text, length=200, overlap_percentage = 0.25):
  """Chunking the text according to the length and overlap_percentage. The percentage overlap is already part of the length """
  overlap_percentage = 1 - overlap_percentage
  total = []
  part = []
  overlap = length - int(length * overlap_percentage) ## number of words in the overlap
  length_new_text = int(length * overlap_percentage) # text length in the transcript that has no overlap with the previous chunk
  if math.ceil(len(text.split())/length_new_text) > 1:
    nr = math.ceil(len(text.split())/length_new_text) ##number of chunks in text according to hyperparameters length and overlap
  else: 
    nr = 1
  # chunking text 
  for w in range(nr):
    if w == 0:
      if nr == 1: # if transcript is less than max_length we do not have to select a part of the transcript
        part = text.split() 
      else: 
        part = text.split()[:length]
      total.append(" ".join(part))
    else:
      part = text.split()[w*length_new_text:w*length_new_text + length_new_text + overlap] 
      total.append(" ".join(part))
 # getting the last remaining part of the transcript that is not yet part of the chunks
  if len(text.split()[(nr-1)*length_new_text + length_new_text + overlap:]) < length: 
    part = text.split()[(nr-1)*length_new_text + length_new_text + overlap:] 
    total.append(" ".join(part))
  return total

import pandas as pd
def iter_chunks(df):
  """Iter function that goes over the df with column textsplit to get the right 
  index and label for each chunk corresponding to the transcript"""
  df_chunk = []
  label_chunk = []
  index_chunk =[]
  for idx,row in df.iterrows():
    for chunk in row['text_split']: ##goes over every chunk in a the list for each transcript
      df_chunk.append(chunk) ## append chunk to list
      label_chunk.append(row[1]) ##append label of chunk/transcript
      index_chunk.append(idx) ## append index of transcript
  return df_chunk, label_chunk, index_chunk

import pandas as pd
def apply_chunking(df, length=200, overlap_percentage = 0.25):
  """Applies the chunking function on a dataframe and returns the new dataframe with the chunked transcripts"""
  df['text_split'] = df['data'].apply(chunks, args=(length, overlap_percentage)) ##splits the transcripts in chunks
 # df.head()
  data_chunk, label_chunk, index_chunk = iter_chunks(df) ## goes over new column with splits in df
  df_new = pd.DataFrame({'data':data_chunk, 'labels':label_chunk, 'index': index_chunk}) #new df with chunks and their labels
  return df_new


from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
def BERT_tokenizer(df, tokenizer, max_len = 512):
  """ Tokenizes each chunk in the dataframe according to the tokenizer rules of BERT and gets the masks"""
  input_ids = []
  transcripts = df['data'].tolist()
  input_ids = np.array([tokenizer.encode(chunk, add_special_tokens=True,max_length=max_len) for chunk in transcripts], dtype=object) # tokenizing 
  input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", value=0, truncating="post", padding="post") ## padding sequences to max length (=length of chunks)
  attention_masks = []
  count = 0
  for chunk in input_ids: ## creating attention masks with 0 and 1 
    att_mask = [int(chunk_id > 0) for chunk_id in chunk]
    if count == 0:
      attention_masks = np.array(att_mask)
      count+=1
    else:
      attention_masks = np.vstack((attention_masks, att_mask))
  return input_ids, attention_masks

import time
import datetime
def elapsed_time(elapsed):
    """"Takes a time in seconds and returns a string hours:minutes:seconds"""
    # time in seconds
    elapsed_rounded = int(round((elapsed)))
    
    # formatting as hours:minutes:seconds
    return str(datetime.timedelta(seconds=elapsed_rounded))

    


import numpy as np
def accuracy(preds, labels):
    """Function to calculate the accuracy of our predictions vs labels"""
    pred = np.argmax(preds, axis=1).flatten()
    label = labels.flatten()
    return np.sum(pred == label) / len(label)


from transformers import AdamW 
from transformers import get_linear_schedule_with_warmup
import numpy as np
from torch import nn
import time
def BERT_model(model, train_dataloader, train_dataloader_noshuffle, val_dataloader, 
               epochs, lr, device):
    """Model for finetuning bert and getting the embeddings. Returns train and 
    validation embeddings and the logits for validation"""
    losses = []
    optimizer = AdamW(model.parameters(), lr = lr)

    # creating a schedule with a learning rate that decreases linearly from the 
    # initial lr set in the optimizer to 0
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0, 
                                                num_training_steps = total_steps)

    print("Finetuning BERT...")
    for epoch_i in range(0, epochs):

        print("")
        print('-------- Epoch {:} / {:} --------'.format(epoch_i + 1, epochs))
        # timing how long the epochs take
        t0 = time.time()
        # resetting the total loss for this epoch
        total_loss = 0
        model.train()
        train_accuracy = 0
        nb_train_steps = 0
        count1 = 0 ## count for concatenating predictions for validation
        for step, batch in enumerate(train_dataloader):
            # progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                #calculating elapsed time in minutes.
                elapsed = elapsed_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, 
                                                                            len(train_dataloader), 
                                                                            elapsed))
            
            # to gpu
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()   
            # feeding inputs, masks and labels to model
            outputs = model(b_input_ids, 
                      attention_mask=b_input_mask,
                      labels=b_labels)
            
            # calculating loss
            loss = outputs[0]
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()

            # clipping to prevent exploding gradients problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # accuracy
            logits = outputs[1] ##logits
            label_ids = b_labels.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            tmp_train_accuracy = accuracy(logits, label_ids)
            train_accuracy += tmp_train_accuracy # per batch

            # track the number of batches
            nb_train_steps += 1
        print('Average loss of epoch' , epoch_i, 'is:', np.mean(losses))
        print("Train accuracy: {0:.2f}".format(train_accuracy/nb_train_steps))

        print("Bert validation")
        t0 = time.time()
        model.eval()
        eval_accuracy = 0
        nb_eval_steps = 0
        count = 0
        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch) # to gpu
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():        
                outputs = model(b_input_ids, 
                                attention_mask=b_input_mask)
                
            # logits for accuracy and auc
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            if count == 0:
                val_logits_list = logits
            else:
                val_logits_list = np.concatenate((val_logits_list, logits))
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            count = 1

            nb_eval_steps += 1
        print("Validation accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))
        print("Validation took: {:}".format(elapsed_time(time.time() - t0)))
        count1 = 0
        
        
    model.eval()
    # getting logits for train set auc
    for batch in train_dataloader_noshuffle:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():        
            outputs = model(b_input_ids, 
                            attention_mask=b_input_mask)
        logits2 = outputs[0]
        logits2 = logits2.detach().cpu().numpy()
        if count1 == 0:
            tr_logits_list = logits2
        else:
            tr_logits_list = np.concatenate((tr_logits_list, logits2))
        count1 = 1
    print("")
    print("Finetuning BERT complete!")
    
    # getting validation embeddings
    model.eval()
    count=0
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():  
            # using underlying BERT model to access last hidden state of cls 
            #token to apply pooled output functions
            outputs = model.distilbert(b_input_ids,
                                       attention_mask=b_input_mask)
# following the object code of distilbertforclassification to get the pooled output (=embeddings)
            pooled_output = outputs[0][:,0] ## index = last hidden state of cls token
            pooled_output = model.pre_classifier(pooled_output)  
            pooled_output = nn.ReLU()(pooled_output)
            pooled_output = pooled_output.detach().cpu().numpy()
            
            # concatenating each batch
            if count == 0:
                v_cls_embeddings = pooled_output
                count += 1
            else:
                v_cls_embeddings = np.concatenate((v_cls_embeddings, pooled_output)) 
    # getting train embeddings            
    model.eval()
    count =0
    for batch in train_dataloader_noshuffle:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():  
            # using underlying BERT model to access last hidden state of cls token to apply pooled output functions
            outputs = model.distilbert(b_input_ids, attention_mask=b_input_mask)
# following the object code of distilbertforclassification to get the pooled output (=embeddings)            
            pooled_output = outputs[0][:,0] ## index = last hidden state of cls token
            pooled_output = model.pre_classifier(pooled_output) 
            pooled_output = nn.ReLU()(pooled_output)  
            pooled_output = pooled_output.detach().cpu().numpy()
            
            # concatenating each batch
            if count == 0:
                t_cls_embeddings = pooled_output
                count += 1
            else:
                t_cls_embeddings = np.concatenate((t_cls_embeddings, pooled_output))
    return t_cls_embeddings, v_cls_embeddings, model, val_logits_list, tr_logits_list

from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
def performance_average_bert(logitslabels, labels_transcripts, index):
  """Getting the BERT mean decision rule performance for each transcript by 
  averaging the results of each chunk"""
  logitslabels2 = np.argmax(logitslabels, axis=1).flatten() ## prediction labels
  x = dict() # creating dictionary with index as keys and predictions as labels 
  # this is done to average the right predictions beloning to the same index/transcript
  for i, l in zip(index, logitslabels2):
    if i in x.keys():
      x[i] =  np.concatenate((x[i], np.array([l])))
    else:
      x[i] = np.array([l])
  ll = []
  for v in x.values():
    ll.append(np.mean(v))
  new = [int(p >= 0.5) for p in ll] # if mean is bigger than 0.5, prediction is 1. Else 0.
  auc_bert = roc_auc_score(labels_transcripts, new)
  acc_bert = accuracy_score(new, labels_transcripts)
  return auc_bert, acc_bert 

# creating new dataset for train and val
import pandas as np
import numpy as np
def RNN_df(embedding, index_list, original_df, length): 
    """creating new dataframe with combined chunk embeddings per transcript 
    with their corresponding label, index and length of the list of embeddings 
    per transcript and looking up the number of the embeddings and max number of 
    embeddings for the padding later in the RNN"""
    x = {}
    embds = []
    # adding the embeddings with same index together (belonging to same transcript)
    for index, emb in zip(index_list, embedding): 
        if index in x.keys():
            x[index] = np.vstack((x[index], emb)) #vstack
        else:
            x[index] = [emb]

    # creating seperate lists for the merged chunks and corresponding labels
    x_final = []
    y_final = []
    for k in x.keys():
        x_final.append(x[k])
        y_final.append(original_df.loc[k]['labels'])
        
    # creating new df
    df = pd.DataFrame({'embedding': x_final, 'labels': y_final, 'index': x.keys()})
    
    # looking up number of chunks per transcript and max number for padding
    length_embs = []
    for embs in df['embedding'].tolist():
      length_embs.append(len(embs))
      max_len = np.max(np.array(length_embs))
    # adding number of chunks as column to be able to sort based on this 
    #for the creation of the batches so that these are created based on their length  
    df['len'] = length_embs
    s = df.len.sort_values(ascending=False).index
    df = df.reindex(s)
    return df, length_embs, max_len

from keras.preprocessing.sequence import pad_sequences
import numpy as np
def padding_emb(embedding, max_len, length_chunks):
  """Padding each transcript according to the maximum number of chunks in is in a transcript"""
  length = len(embedding)
  padding_len = max_len - length
  for i in range(padding_len):
    embedding = np.vstack((embedding, np.zeros(768))) # 768 is length of each embedding 
    # or length of parameters in last layer of BERT model
  return embedding

import torch.optim 
import torch.nn as nn
import numpy as np
class Recurrent(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, recurrent):
    super().__init__()
    self.recurrent = recurrent(input_dim, hidden_dim, num_layers = num_layers, batch_first=True, dropout=dropout)
    self.linear = nn.Linear(hidden_dim, output_dim)
  def forward(self, text, seq_len):
    # packing for masking
    packed = torch.nn.utils.rnn.pack_padded_sequence(text, seq_len.cpu().numpy(), batch_first=True, enforce_sorted=True)
    output, h_n = self.recurrent(packed)
    if type(self.recurrent) == nn.LSTM:
      last = h_n[0][-1,:,:] ## lstm also returns memory cell output at h_n[1]. We don't want that
    else:
      last = h_n[-1,:,:]
    return self.linear(last)

from sklearn.metrics import roc_auc_score
import torch.optim 
import torch.nn as nn
import numpy as np
def train_epoch(model, dataloader, optimizer, criterion, device):
  """Train model in batches from dataloader and return training loss and 
  auc."""
  epoch_loss = 0
  epoch_acc = 0 
  c = 0 # count for concatenating batches for predictions and labels (auc)

  model.train()
  for batch in dataloader:
    data, label, seq_len = batch
    # to gpu
    data, label, seq_len = data.to(device), label.to(device), seq_len.to('cpu')

    optimizer.zero_grad()
    pred = model(data, seq_len).squeeze(1) # predictions
    loss = criterion(pred, label) # calculating loss
    loss.backward()
    optimizer.step()

    # using labels and predictions for AUC
    pred = pred.detach().cpu().numpy()
    prediction = [int(p >= 0.0) for p in pred]
    label = label.cpu()
    if c == 0:
      predictions = prediction
      labels = label
      c+= 1
    else:
      predictions = np.concatenate((predictions, prediction))
      labels = np.concatenate((labels, label))

    epoch_loss += loss.item()
  return epoch_loss / len(dataloader), roc_auc_score(labels, predictions)

import torch.optim 
import torch.nn as nn
import numpy as np
def evaluate(model, dataloader, criterion, device):
  """Evaluate model on all batches from dataloader."""
  loss = 0
  acc = 0
  c = 0
  labellist = []
  model.eval()
  with torch.no_grad():
    for batch in dataloader:
      data, label, seq_len = batch
      # to gpu
      data, label, seq_len = data.to(device), label.to(device), seq_len.to('cpu')
      pred = model(data, seq_len).squeeze(1) # predictions
      loss += criterion(pred, label).item() # calculating loss
      # getting predictions and labels for auc score
      pred = pred.detach().cpu().numpy()
      prediction = [int(p >= 0.0) for p in pred]
      label = label.cpu()
      if c == 0:
        predictions = prediction
        labels = label
        c+= 1
      else:
        predictions = np.concatenate((predictions, prediction))
        labels = np.concatenate((labels, label))
    return loss/len(dataloader), roc_auc_score(labels, predictions)

import logging
import torch.optim 
import torch.nn as nn
import numpy as np
logging.basicConfig(level=logging.INFO)
def train_RNN(model, train_it, val_it, pos_weight, optimizer, epochs, device):
  """Training RNN model"""
  # setting loss function
  criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.array([pos_weight])).cuda())
  history = dict(auc=[], val_auc=[], loss=[], val_loss=[])
  logging.info("Epoch Loss Val_loss Auc Val_auc")
  device = 'cuda'
  # training RNN and getting auc and loss for both train and vall
  for epoch in range(1, epochs+1):
    optimizer.zero_grad()
    loss, auc = train_epoch(model, train_it, optimizer, criterion, device)
    history['auc'].append(auc)
    history['loss'].append(loss)
    val_loss, val_auc = evaluate(model, val_it, criterion, device)
    history['val_auc'].append(val_auc)
    history['val_loss'].append(val_loss)
    logging.info(f"{epoch:3d} {loss:.3f} {val_loss:.3f} {auc:.3f} {val_auc:.3f}")
  last_tr_auc = history['auc'][-1] # last auc from last epoch
  last_val_auc = history['val_auc'][-1] # last auc from last epoch
  return history, last_tr_auc, last_val_auc

import pandas as pd
def remover_empty_rows(df):
    """Removing empty rows that are a result of chunking"""
    condition = df.data.apply(lambda x: len(x.split())) > 1
    #print(sum(condition==False))
    df = df[condition]
    return df

from torch import nn
import numpy as np
def random_search_param():
  """Random search function for RoBERT"""
  length_chunks = np.random.randint(100,501)
  percentage_overlap = np.round(np.random.uniform(0.1, 0.4),2)
  bert_lr = np.random.uniform(5e-5, 1e-5) #based upon author's recommendation (Devlin et al. 2019)
  bert_epochs = np.random.choice(np.array([2, 3, 4])) #based upon author's recommendation (Devlin et al. 2019)
  batch_size_RNN = np.random.randint(1, 32)
  pos_weight = np.random.uniform(1, 4)
  RNN_type = np.random.choice(np.array([nn.RNN, nn.LSTM, nn.GRU]))
  RNN_epochs = np.random.randint(5, 50)
  RNN_lr = np.round(np.random.uniform(0.000001, 0.001),6)
  RNN_wd = np.round(np.random.uniform(0, 0.001),6)
  RNN_layers = np.random.randint(1, 3)
  RNN_units = np.random.randint(5, 101)
  dropout = np.round(np.random.uniform(0, 0.5),2)
  return length_chunks, percentage_overlap, bert_lr, bert_epochs, batch_size_RNN, pos_weight, RNN_type, RNN_epochs, RNN_lr, RNN_wd, RNN_layers, RNN_units, dropout

import random
import torch
import pandas as pd
import torch
import random
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertModel, DistilBertForSequenceClassification
import csv
def hyperparametertuning(train_df, val_df, tokenizer, nr_jobs):
  """Hyperparameter tuning of RoBERT"""
  results = []
  for i in range(nr_jobs):
    length_chunks, percentage_overlap, bert_lr, bert_epochs, batch_size_RNN, pos_weight,RNN_type, RNN_epochs, RNN_lr, RNN_wd, RNN_layers, RNN_units, dropout = random_search_param()
    performance = {'length_chunks': length_chunks, 'percentage_overlap': percentage_overlap, 'bert_lr': bert_lr, 
                  'bert_epochs': bert_epochs, 'batch_size_RNN': batch_size_RNN, 'RNN_epochs': RNN_epochs,
                  'pos_weight': pos_weight,'RNN_type': RNN_type, 'RNN_lr': RNN_lr, 'RNN_wd': RNN_wd, 
                  'RNN_layers': RNN_layers, 'RNN_units': RNN_units, 'dropout': dropout}
    print(f"Nr_jobs = {i+1}, using the following hyperparameters: \n {performance}")

    # chunking
    print("Chunking...")
    df_new_train = apply_chunking(train_df, length_chunks, percentage_overlap)
    df_new_val = apply_chunking(val_df, length_chunks, percentage_overlap)
    print("Chunking done.")
    # removing empty rows
    df_new_train = remover_empty_rows(df_new_train)
    df_new_val = remover_empty_rows(df_new_val)
    print()
    print(df_new_train.head())
    print(df_new_val.head())
    print()
    # tokenizing and mapping to input ID's 
    print("Tokenizing...")
    train_inputIDs, train_masks = BERT_tokenizer(df_new_train, tokenizer, max_len = length_chunks)
    val_inputIDs, val_masks = BERT_tokenizer(df_new_val, tokenizer, max_len = length_chunks)
    print("Tokenizing done.")
    print()
    
    ### BERT
    print("Transforming into tensors...")
    train_labels = df_new_train['labels'].tolist()
    train_inputs = torch.LongTensor(train_inputIDs)
    train_labels = torch.LongTensor(train_labels)
    train_masks2 = torch.FloatTensor(train_masks)
    #print(train_inputs.shape, train_labels.shape, train_masks2.shape)
    train_data = TensorDataset(train_inputs, train_masks2, train_labels)
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle = True) 
    train_dataloader_nosshuffle = DataLoader(train_data, batch_size=16, shuffle = False)

    val_labels = df_new_val['labels'].tolist()
    val_inputs = torch.LongTensor(val_inputIDs)
    val_labels = torch.LongTensor(val_labels)
    val_masks2 = torch.FloatTensor(val_masks)
    val_data = TensorDataset(val_inputs, val_masks2, val_labels)
    val_dataloader = DataLoader(val_data, batch_size=16)
    print("Transforming done.")
    print()

    # getting embeddings from BERT
    print("finetuning BERT and getting BERT embeddings...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1_bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', output_hidden_states=True).to(device)
    model1_bert.resize_token_embeddings(len(tokenizer)) 
    tr_emb, val_emb, model, val_logits_list, tr_logits_list = BERT_model(model1_bert, train_dataloader, train_dataloader_nosshuffle, val_dataloader, epochs = bert_epochs, 
                                    lr = bert_lr, device = device) 
    print("Getting BERT embeddings done.")

    ## performance of BERT mean decision rule
    val_auc_bert, val_acc_bert = performance_average_bert(val_logits_list, val_df['labels'].tolist(), df_new_val['index'].tolist())
    tr_auc_bert, tr_acc_bert = performance_average_bert(tr_logits_list, train_df['labels'].tolist(), df_new_train['index'].tolist())
    performance['val_auc_bert'] = val_auc_bert
    performance['val_acc_bert'] = val_acc_bert
    performance['tr_auc_bert'] = tr_auc_bert
    performance['tr_acc_bert'] = tr_acc_bert
    print()
    ### RNN
    print('Creating new dataframes with embeddings per transcript for RNN...')
    train_df_RNN, tr_length_embs, tr_emb_maxlen = RNN_df(tr_emb, df_new_train['index'].tolist(), train_df, length_chunks) 
    val_df_RNN, tr_length_embs, val_emb_maxlen, = RNN_df(val_emb, df_new_val['index'].tolist(), val_df, length_chunks)
    print(train_df_RNN.head())
    print(val_df_RNN.head())
    print('Done.')
    print()

    ## padding data
    train_df_RNN['padded_data'] = train_df_RNN['embedding'].apply(padding_emb, args=(tr_emb_maxlen, length_chunks))
    train_inputs2 = train_df_RNN['padded_data'].to_list()
    train_labels2 = train_df_RNN['labels'].to_list()
    
    # transforming data into tensors
    seq_list_tr = torch.tensor(train_df_RNN['len'].tolist()) # to let RNN know 
    #to pay attention from index 0 to length of transcripts and to ignore zero's that come after index 'length'
    train_inputs2 = torch.FloatTensor(train_inputs2)
    train_labels2 = torch.FloatTensor(train_labels2)
    train_data2 = TensorDataset(train_inputs2,train_labels2, seq_list_tr)
    train_dataloader2 = DataLoader(train_data2, batch_size=int(batch_size_RNN))

    # same for validation set: padding, tensors and sequence list for masking
    val_df_RNN['padded_data'] = val_df_RNN['embedding'].apply(padding_emb, args=(val_emb_maxlen, length_chunks))
    val_inputs2 = val_df_RNN['padded_data'].to_list()
    val_labels2 = val_df_RNN['labels'].to_list()
    seq_list_val = torch.tensor(val_df_RNN['len'].tolist()) #list of number of embeddings for each transcript
    val_inputs2 = torch.FloatTensor(val_inputs2)
    val_labels2 = torch.FloatTensor(val_labels2)
    val_data2 = TensorDataset(val_inputs2, val_labels2, seq_list_val)
    val_dataloader2 = DataLoader(val_data2, batch_size=len(val_inputs2))

    input_dim = train_inputs2.shape[2]
    print(" Training RNN...")
    model2 = Recurrent(recurrent = RNN_type, input_dim=input_dim, hidden_dim = RNN_units, num_layers=RNN_layers, output_dim=1, dropout = dropout).to('cuda') #RNN_layers
    optimizer = torch.optim.Adam(model2.parameters(), lr = RNN_lr, weight_decay= RNN_wd)
    history, tr_auc, val_auc = train_RNN(model2, train_dataloader2 ,val_dataloader2, pos_weight, optimizer, epochs=RNN_epochs, device = device)
    print("RNN done.")
    performance['val_AUC'] = val_auc
    performance['tr_AUC'] = tr_auc
    print()
    #print('Hyperparameters and performance:', performance)
    results.append(performance)
    print(results)
    toCSV = results
    keys = toCSV[0].keys() ## appending results to csv to make sure results are not lost
    with open('/content/drive/My Drive/Thesis/RoBERT_results.csv', 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)
  highest_robert = sorted(results,  key=lambda x: x['val_auc_bert'], reverse = True )[0] # highest performance robert
  highest_bert = sorted(results,  key=lambda x: x['val_AUC'], reverse = True )[0] # highest performance BERT
  print('highest robert:', highest_robert)
  print('highest bert:', highest_bert)
  return results, highest_robert, highest_bert



