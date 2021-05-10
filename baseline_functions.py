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
from sklearn.metrics import roc_auc_score
def train_epoch(model, iterator, optimizer, criterion):
    """Train `model` in batches from `iterator` and return training loss and 
    AUC score."""
    epoch_loss = 0
    c = 0 

    model.train() 
    for batch in iterator:
    #print(batch)
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

    return epoch_loss / len(iterator), roc_auc_score(labels, predictions)

import numpy as np
from sklearn.metrics import roc_auc_score
def evaluate(model, iterator, criterion):
    """Evaluate `model` on all batches from `iterator`."""
    loss = 0
    c = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            pred = model(batch.data).squeeze(1)
            loss += criterion(pred, batch.labels).item()

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

    return loss/len(iterator), roc_auc_score(labels, predictions)

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