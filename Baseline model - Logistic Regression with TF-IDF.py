"""
Created on Fri Apr  2 18:19:43 2021

@author: Ivette Bonestroo
"""

import pickle
from baseline_functions import preprocessing2, remover, random_search_LR
from data_functions import pickle_splitter, binarizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import numpy as np
import csv

path = 'training_data.pickle'
path2 = 'test_data.pickle'

with open(path, 'rb') as file:
    train = pickle.loads(file.read())

with open(path2, 'rb') as file:
    test = pickle.loads(file.read())

## merging labels and transcripts together and merging the labels 2 and 3 (conspiracy labels)
transtr, labeltr = pickle_splitter(train)
transtest, labeltest = pickle_splitter(test)
y_tr = binarizer(labeltr)
y_test = binarizer(labeltest)

total_labels_old = labeltr + labeltest
print('labels before removing')
print(f'number of labels 1: {labeltr.count(1) + labeltest.count(1)}')
print(f'number of labels 2: {labeltr.count(2) + labeltest.count(2)}')
print(f'number of labels 3: {labeltr.count(3) + labeltest.count(3)}')

## creating a df for labelcount before binarizing
labels3_train_df = pd.DataFrame(transtr, columns = ['data'])
labels3_train_df['labels'] = labeltr
labels3_test_df = pd.DataFrame(transtest, columns = ['data'])
labels3_test_df['labels'] = labeltest

## removing transcripts containing a very few words
print(f"length train: {len(labels3_train_df)}, length test: {len(labels3_test_df)}")
print('Removing incomplete transcripts..')
labels3_train_df = remover(labels3_train_df, 50)
labels3_test_df= remover(labels3_test_df, 50)
print(f"length train: {len(labels3_train_df)}, length test: {len(labels3_test_df)}")

train_3labels = labels3_train_df['labels'].tolist()
test_3labels = labels3_test_df['labels'].tolist()
## labels after removing
print(f'number of labels 1: {train_3labels.count(1) + test_3labels.count(1)}')
print(f'number of labels 2: {train_3labels.count(2) + test_3labels.count(2)}')
print(f'number of labels 3: {train_3labels.count(3) + test_3labels.count(3)}')
total_labels_old = train_3labels + test_3labels

train_df = pd.DataFrame(transtr, columns = ['data'])
train_df['labels'] = y_tr
test_df = pd.DataFrame(transtest, columns = ['data'])
test_df['labels'] = y_test

print(test_df.head())
print(train_df.head())
print()

## removing transcripts containing a very few words
#print(f"length train: {len(train_df)}, length test: {len(test_df)}")
#print('Removing incomplete transcripts..')
train_df = remover(train_df, 50)
test_df= remover(test_df, 50)
print(f"length train: {len(train_df)}, length test: {len(test_df)}")

train2labels = train_df['labels'].tolist()
test2labels = test_df['labels'].tolist()
print('labels after binarizing and removing')
print(f'number of labels 0: {train2labels.count(0) + test2labels.count(0)}')
print(f'number of labels 1: {train2labels.count(1) + test2labels.count(1)}')

trainval_df = train_df.copy()
## creating validation set
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(train_df, test_size=0.25, 
                                    stratify = train_df['labels'], random_state = 2021)
print()
print('length train df, val df and test df')
print(len(train_df), len(val_df), len(test_df))


# checking punctuation in transcripts
punctuation_tr = []
for t in train_df['data']:
    if ',' in t or '.' in t:
        punctuation_tr.append(0)
    else:
        punctuation_tr.append(1)

punctuation_val = []
for t in val_df['data']:
    if ',' in t or '.' in t:
        punctuation_val.append(0)
    else:
        punctuation_val.append(1)
        
train_df['punctuation'] = punctuation_tr
val_df['punctuation'] = punctuation_val
train_val = pd.concat([train_df, val_df])

print(punctuation_tr)
print(punctuation_val)

## graphs
# target distribution graph 3 labels
ax = sns.countplot(x = total_labels_old, hue = total_labels_old, dodge = False)
h,l = ax.get_legend_handles_labels()
labels = ['Non-conspiracy', 'Falsifiable conspiracy', 'Unfalsifiable conspiracy']
ax.legend(h,labels,title="Label", loc="upper right") 
plt.xlabel('Labels')
plt.ylabel('Count')
#plt.title('Train+val: Target distribution')

# graph of target distribution 2 labels
ax = sns.countplot(x = train_val['labels'], hue = train_val['labels'], dodge = False)
h,l = ax.get_legend_handles_labels()
labels = ['Non-conspiracy', 'Conspiracy']
ax.legend(h,labels,title="Label", loc="upper right") 
plt.xlabel('Labels')
plt.ylabel('Count')
#plt.title('Train+val: Target distribution')

# punctuation graph
ax = sns.countplot(x=train_val['punctuation'], hue=train_val['punctuation'], dodge=False)
h,l = ax.get_legend_handles_labels()
labels = ['With punctuation', 'Without punctuation']
ax.legend(h,labels,title="Punctuation", loc="upper right") 
#plt.title('Train+val: punctuation distribution')
plt.xlabel('Punctuation')
plt.ylabel('Count')

# graph of length
length_t = sorted([len(t.split()) for t in train_val['data']])
#print(sum(length_t)/len(length_t))
plt.plot(length_t)
#plt.ylim(bottom=min(length_t))  
#plt.yticks(np.arange(min(length_t), max(length_t),5000)) 
#plt.title('Transcript word length')
plt.xlabel('Transcripts')
plt.ylabel('Word length')
print(min(length_t))
print(max(length_t))


def logistic_regression_tuning():
    max_features = random_search_LR()
    performance = {'max_features': max_features}
    TV = TfidfVectorizer(tokenizer=preprocessing2, max_features = max_features)
    X_train = TV.fit_transform(train_df['data'].tolist())
    X_val = TV.transform(val_df['data'].tolist())
    
    ## logistic regression 
    logreg = LogisticRegression(random_state = 2021)
    logreg.fit(X_train, train_df['labels'])
    preds = logreg.predict(X_val)

    ## roc_auc_score
    auc = roc_auc_score(val_df['labels'].tolist(), preds)
    performance['auc'] = auc
    return performance

#hyperparametertuning
results = []
for i in range(26):
    performance = logistic_regression_tuning()
    print(performance)
    results.append(performance)
print(results)

# saving results of hyperparameter tuning
toCSV = results
keys = toCSV[0].keys()
with open('LR_results.csv', 'w', newline='')  as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(toCSV)
highest = sorted(results,  key=lambda x: x['auc'], reverse = True )[0]
print(highest)
best_max_features = highest['max_features']

# training with train+val
TV = TfidfVectorizer(tokenizer=preprocessing2, max_features = best_max_features)
X_trainval = TV.fit_transform(trainval_df['data'].tolist())
X_test = TV.transform(test_df['data'].tolist())

## logistic regression evaluation
logreg = LogisticRegression(random_state = 2021)
logreg.fit(X_trainval, trainval_df['labels'])
preds = logreg.predict(X_test)
val_auc = roc_auc_score(test_df['labels'].tolist(), preds)
print('validation auc', val_auc) #0.8281959766385465

#training auc
preds = logreg.predict(X_trainval)
train_auc = roc_auc_score(trainval_df['labels'].tolist(), preds)
print('training auc', train_auc) #0.9275635930047695


