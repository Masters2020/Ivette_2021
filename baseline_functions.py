# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 10:17:38 2021

@author: Ivette Bonestroo
"""
### preprocessing code based on code of Omer Ahmed's thesis (2020)
def preprocessing2(transcript):
    import re
    import nltk

  ## remove words between [] and ()
    new_stopwords = []
    z = re.findall("(\[.*?\])", transcript)
    x = re.findall("(\(.*?\))", transcript)
    for word in z:
        new_stopwords.append(word)
    for word in x:
        new_stopwords.append(word)
    
    ## common stopwords
    new_stopwords.extend(['Music', 'music', 'MUSIC', 'Applause',  'APPLAUSE', 'applause']) 
    #print(new_stopwords)
    new_stopwords = list(set(new_stopwords))
    
    for word in new_stopwords:
        if len(word.split()) > 10:
            continue
        if word in transcript:
            transcript = transcript.replace(word, "")
    transcript = transcript.replace("\n", " ") ## replacing inconsistencies with a space
    transcript = transcript.replace("xa0", " ")
    #removes non-alpha numeric
  
    b = []
    x = re.findall("[^0-9A-Za-z ]", transcript)
    for char in x:
        b.append(char)
    nonalphanum = list(set(b))


    words = nltk.word_tokenize(transcript)
    for word in words:
        if word in nonalphanum:
            words.remove(word)
            

#----------------------------------------------------------#
# lemmatize the document. 
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer 
    lemmatizer = WordNetLemmatizer()

    lemmatized = [] # to store lematized transcript

    for word in words:

        tag_tuple = nltk.pos_tag([word])[0][1]
        tag = tag_tuple[0] # takes out the tag for each word
        # lemmatize the word with the appropriate tag
        if tag == 'V':
            lemmatized.append(lemmatizer.lemmatize(word, wordnet.VERB))
        elif tag =="N":
            lemmatized.append(lemmatizer.lemmatize(word, wordnet.NOUN))
        elif tag =="J":
            lemmatized.append(lemmatizer.lemmatize(word, wordnet.ADJ))    
        elif tag =="R":
            lemmatized.append(lemmatizer.lemmatize(word, wordnet.ADV))

#----------------------------------------------------------#
#remove stop words

    from nltk.corpus import stopwords

    stop_words = stopwords.words('english')
    stop_words.append('me')
    no_stop_word = [] # vector to store all sentences after processing 

    for word in lemmatized:
        if word not in stop_words:
            no_stop_word.append(word)


#----------------------------------------------------------#

#### Remove any spaces(if any)
### Lower case everything

    transcript_vector=[] # to store words for 1 transcript
    for word in no_stop_word:
        if len(word)> 1:
            transcript_vector.append(word.lower()) # appends the remaining words, lower cased, in this vector

    return transcript_vector

import pandas as pd
def remover(df, length=200):
    """Remove transcripts that have less than length words"""
    condition = df.data.apply(lambda x: len(x.split())) >= length
#  print(sum(condition==False))
    df = df[condition]
    return df

## recurrent model class
from torch import nn
import torch
class Recurrent(nn.Module):
    def __init__(self, embedding, input_dim, embedding_dim, hidden_dim, num_layers, output_dim, 
                 recurrent, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.recurrent = recurrent(embedding_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_dim, output_dim)
    def forward(self, text):
        embedded = self.embedding(text)
        output, h_n = self.recurrent(embedded)
        if type(self.recurrent) == nn.LSTM:
            last = h_n[0][-1,:,:]
        else:
            last = h_n[-1,:,:]
        return self.linear(last)

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
def train_epoch(model, iterator, optimizer, criterion):
    """Train `model` in batches from `iterator` and return training loss and 
    AUC score."""
    epoch_loss = 0
    c = 0 

    model.train() 
    for batch in iterator:
        optimizer.zero_grad()
        pred = model(batch.data).squeeze(1)

        loss = criterion(pred, batch.labels)
        pred = pred.detach().cpu().numpy()

        ## concatenating all predictions and labels from each batch for AUC
        prediction = [int(p >= 0.0) for p in pred]
        label = batch.labels.cpu()
        if c == 0:
            predictions = prediction
            labels = label
            c+= 1
        else:
            predictions = np.concatenate((predictions, prediction)) 
            labels = np.concatenate((labels, label)) 

        loss.backward()
        ## clipping the gradients to prevent exploding/vanishing gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator), roc_auc_score(labels, predictions), accuracy_score(labels,predictions)

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
def evaluate(model, iterator, criterion):
    """Evaluate `model` on all batches from `iterator`."""
    loss = 0
    c = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            pred = model(batch.data).squeeze(1) #getting predictions
            loss += criterion(pred, batch.labels).item() #loss function

            ## concatenating all predictions and labels from each batch for AUC
            label = batch.labels.cpu()
            pred = pred.detach().cpu().numpy()
            prediction = [int(p >= 0.0) for p in pred]
            if c == 0:
                predictions = prediction
                labels = label
                c+= 1
            else:
                predictions = np.concatenate((predictions, prediction))
                labels = np.concatenate((labels, label))

        return loss/len(iterator), roc_auc_score(labels, predictions), accuracy_score(labels,predictions)

import numpy as np
def random_search_LR():
    """Random search function for logistic regression"""
    max_features = np.random.randint(1000,10001)
    return max_features

import numpy as np
def random_search_RNN():
    """Random search function for RNN"""
    batch_size_RNN = np.random.randint(1, 32)
    pos_weight = np.random.uniform(1, 4)
    RNN_type = np.random.choice(np.array([nn.RNN, nn.LSTM, nn.GRU]))
    RNN_epochs = np.random.randint(1, 31)
    RNN_lr = np.round(np.random.uniform(0.000001, 0.05),6)
    RNN_wd = np.round(np.random.uniform(0, 0.0001),6)
    RNN_layers = 1
    RNN_units = np.random.randint(5, 51)
    dropout = np.round(np.random.uniform(0, 0.5),2)
    return batch_size_RNN, pos_weight, RNN_type, RNN_epochs, RNN_lr, RNN_wd, RNN_layers, RNN_units, dropout

import numpy as np 
import torch 
from torch import nn
import torchtext 
import torchtext.legacy.data 
from torchtext import vocab
import csv
import logging
logging.basicConfig(level=logging.INFO)
def hyperparametertuning(train, val, nr_jobs, TEXT, pretrained_embeddings, 
                         input_dim, embedding_dim = 300):
  """Function for the hyperparameter tuning of the RNN glove model"""
  results = []
  for i in range(nr_jobs):
    # getting random hyperparameter settings
    batch_size_RNN, pos_weight,RNN_type, RNN_epochs, RNN_lr, RNN_wd, RNN_layers, RNN_units, dropout = random_search_RNN()
    performance = {'batch_size_RNN': batch_size_RNN, 'RNN_epochs': RNN_epochs,
                  'pos_weight': pos_weight,'RNN_type': RNN_type, 'RNN_lr': RNN_lr, 'RNN_wd': RNN_wd, 
                  'RNN_layers': RNN_layers, 'RNN_units': RNN_units, 'dropout': dropout}
    print(f"Nr = {i+1}, using the following hyperparameters: \n {performance}")
    # splitting train and val in batches
    train_iter = torchtext.legacy.data.BucketIterator(train, batch_size=batch_size_RNN,
                                                      sort_key=lambda x: len(x.data),
                                                      sort_within_batch=False,
                                                      device='cuda') ##TES
    val_iter = torchtext.legacy.data.BucketIterator(val, batch_size=batch_size_RNN,
                                                    sort_key=lambda x: len(x.data),
                                                    sort_within_batch=False,
                                                    device='cuda') ##TEST
    input_dim = input_dim
    embedding_dim = embedding_dim
    model = Recurrent(recurrent = RNN_type, input_dim=input_dim, embedding_dim = embedding_dim, 
                      hidden_dim = RNN_units, num_layers=RNN_layers, output_dim=1, dropout = dropout, embedding = vocab.Vectors).cuda() #RNN_layers
    
    ## updating model embedding with glove embedding weights
    model.embedding.weight.data = pretrained_embeddings.cuda() 

    unknown_index = TEXT.vocab.stoi[TEXT.unk_token] # get index of unknown token
    padding_index = TEXT.vocab.stoi[TEXT.pad_token] # get index of padding token
    model.embedding.weight.data[unknown_index] = torch.zeros(embedding_dim) #change values to zeros
    model.embedding.weight.data[padding_index] = torch.zeros(embedding_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr = RNN_lr , weight_decay= RNN_wd)
    history = dict(auc=[], val_auc=[], loss=[], val_loss=[], acc=[], val_acc=[])

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(np.array([pos_weight]))).cuda()
    logging.basicConfig(level=logging.INFO)
    logging.info("Epoch Loss Val_loss Auc Val_auc Acc Val_acc")
    ## getting val and train loss & auc
    for epoch in range(1, RNN_epochs+1):
      optimizer.zero_grad()
      loss, auc, acc = train_epoch(model, train_iter, optimizer, criterion) #training
      history['auc'].append(auc)
      history['loss'].append(loss)
      history['acc'].append(acc)
      val_loss, val_auc, val_acc = evaluate(model, val_iter, criterion) #evaluating
      history['val_auc'].append(val_auc)
      history['val_loss'].append(val_loss)
      history['val_acc'].append(val_acc)
      logging.info(f"{epoch:3d} {loss:.3f} {val_loss:.3f} {auc:.3f} {val_auc:.3f}  {acc:.3f} {val_acc:.3f}")
    last_val_auc = history['val_auc'][-1] # last auc of last epoch
    last_tr_auc = history['auc'][-1]
    last_val_acc = history['val_acc'][-1] # last acc of last epoch
    last_tr_acc = history['acc'][-1]
    performance['val_auc'] = last_val_auc
    performance['tr_auc'] = last_tr_auc
    performance['val_acc'] = last_val_acc
    performance['tr_acc'] = last_tr_acc
    results.append(performance)
    toCSV = results
    keys = toCSV[0].keys() # saving results
    with open('RNN_glove.csv', 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(toCSV)
    #print(results)
  highest = sorted(results,  key=lambda x: x['val_auc'], reverse = True )[0] # hyperparameters with highest performance
 # print(highest)
  return results, highest
