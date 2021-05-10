# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:36:49 2021

@author: Ivette-PC
"""
def pickle_splitter(pickle):
  """Splits the loaded pickle in data and labels"""
  trans = []
  label = []
  for n in pickle:
      trans.append(n[0])
  for n in pickle:
      label.append(n[1])      
  return trans, label

def binarizer(labels):
  """Binarizes the labels"""
  y = []
  for document in labels:
      class_made = 0
      if document == 1:
          class_made = 0
      else:
          class_made = 1
      y.append(class_made)
  return y