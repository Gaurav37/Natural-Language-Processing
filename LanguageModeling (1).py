#!/usr/bin/env python
# coding: utf-8

# Training *character-level* language models. 
# We will train unigram, bigram, and trigram character-level models on a collection of books from Project Gutenberg. We will then use these trained English language models to distinguish English documents from Brazilian Portuguese documents in the test set.

# In[28]:


import pandas as pd
import httpimport
import sklearn_pandas
from sklearn.model_selection import train_test_split
with httpimport.remote_repo(['lm_helper'], 'https://raw.githubusercontent.com/jasoriya/CS6120-PS2-support/master/utils/'):
  from lm_helper import get_train_data, get_test_data


# In[55]:


# get the train and test data
train = get_train_data()
test, test_files = get_test_data()
train,held_out  = train_test_split(train, test_size=0.20)


# Collecting some statistics on the unigram, bigram, and trigram character counts.

# In[31]:


import re
def sentences(train):
  train_edit=[]
  for i in train:
    for j in i:
      sentence=" ".join(j)
      sentence=sentence.lower()
      sentence = re.sub(r'[^A-Za-z. ]', '', sentence)
      train_edit.append(sentence)
  return train_edit

train_edit=sentences(train)
print("train_edit",train_edit[0])
########################################
def grams(n,train_edit):
  gram_set=[]
  for i in train_edit:
    for j in range(len(i)-n+1):
      if n==1:
        gram_set.append(i[j:j+n])
      elif n==2:
        gram_set.append(i[j:j+n])
      else:
        gram_set.append(i[j:j+n])
  return gram_set


# In[32]:


unigram=grams(1,train_edit)
bigram=grams(2,train_edit)
trigram=grams(3,train_edit)

for a in range(0,10):
  print(trigram[a])
# doing well


# In[33]:


def stats(grams):
  dict={}
  for item in grams:
    if (item in dict):
      dict[item] += 1
    else:
      dict[item] = 1
  return dict


# In[34]:


unigram_dict=stats(unigram)
bigram_dict=stats(bigram)
trigram_dict=stats(trigram)


# In[35]:


print(trigram_dict.keys())
if 'man' in trigram_dict:
  print("yes")


# In[36]:


import numpy as np

lambda_val=[[0.7,0.2,0.0999,0.0001],
            [0.7,0.15,0.1499,0.0001],
            [0.8,0.1,0.0999,0.0001],
            [0.6,0.2,0.1999,0.0001],
            [0.82,0.1,0.0799,0.0001],
            [0.85,0.1,0.0499,0.0001],
            [0.85,0.12,0.0299,0.0001],
            [0.9,0.05,0.0499,0.0001],
            [0.9,0.08,0.0199,0.0001],
            [0.95,0.03,0.0199,0.0001]]
for i in lambda_val:
  print(np.sum(i))
print(np.sum(lambda_val, axis=1))


# In[42]:


import math
def perplexity_with_interpolation(lamb,sentence):
  probs=0.0
  count=0
  for i in range(len(sentence)-2):
    token= sentence[i:i+3]
    max_likelyhood=0
    if token in trigram_dict.keys():
      max_likelyhood+=lamb[0]*trigram_dict[token]/bigram_dict[token[:-1]]
    token =token[:-1]
    if token in bigram_dict.keys():
      max_likelyhood+=lamb[1]*bigram_dict[token]/unigram_dict[token[:-1]]
    token=token[:-1]
    if token in unigram_dict.keys():
      max_likelyhood+= lamb[2]*unigram_dict[token]/(sum(unigram_dict.values()))
    else:
      max_likelyhood+= lamb[3]/(sum(unigram_dict.values()))
    max_likelyhood=math.log(max_likelyhood)#log of likelyhood after interpolating linearly with 4 lambdas lamb
    probs=probs + max_likelyhood
    count+=1
  perplexity=2**(-1*probs/count)
  return perplexity  


# In[50]:


held_out_sentences=sentences(held_out)
print("sentences", held_out_sentences)
#print("".join(held_out_sentences))


# In[48]:


#Best Lambda
Perp_val_for_lamb=[]
for i in lambda_val:
  Perp_val=perplexity_with_interpolation(i,"".join(held_out_sentences))
  print(i," Perplexity with this lambda set is : ", Perp_val)
  Perp_val_for_lamb.append(Perp_val)
best_lambda=lambda_val[Perp_val_for_lamb.index(min(Perp_val_for_lamb))]
print("Best lambda value set: ",best_lambda)


# In[123]:


print(len(test))
print(len(test_files),len(test_sentences))
test_data=[]
for i in range(0,len(test)):
  total=[]
  for j in test[i]:
    total += j
  total=" ".join(total)
  total= re.sub(r'[^A-Za-z. ]', '', total) 
  test_data.append(total)

for i in range(0,10):
  print(test_data[i])
  print(len(test_data))


# In[85]:


#test_sentences=sentences(test)
#test_data="".join(held_out_sentences)
test_file_perp={}
for i in range(0,len(test_data)):
  Perp_val=perplexity_with_interpolation(best_lambda,test_data[i])
  test_file_perp[test_files[i]]=Perp_val
  #print(test_files[i]," : ", Perp_val)
sorted_perp_test_files = dict( sorted(test_file_perp.items(),
                           key=lambda item: item[1],
                           reverse=False))
#print('Sorted Dictionary: ')
print(sorted_perp_test_files)


# In[103]:


print(test_data[test_files.index("ce09")])
#threshold=9.09, higher or equal to 9.09 is Brazilian file


# Perplexity for each document in the test set using:
# linear interpolation smoothing method. 
# For determining λs for linear interpolation, we have divided training data into training data and held_out data(20%), then using grid search method:
# Choosing ~10 values of λ to test using grid search on held-out data.
# 
# Then determining the accuracy by correct files identified where we know that out of all files provided , the files ending with .txt are Brazilian. Print the file names (from `test_files`) and perplexities of the documents above the threshold
# 
#     ```
#         file name, score
#         file name, score
#         . . .
#         file name, score
#     ```

# In[105]:


# accuracy
threshold_val_perp=9.09
def accuracy(sorted_perp_test_files,threshold_val_perp):
  correct=0
  wrong=0
  for k,v in sorted_perp_test_files.items():
    if v >threshold_val_perp:
      if re.search("txt",k):
        correct+=1
      else:
        wrong+=1
    else:
      if re.search("txt",k):
        wrong+=1
      else:
        correct+=1
  acc=round((correct*100)/(correct+wrong),2)
  return acc


# In[106]:


print(accuracy(sorted_perp_test_files,9.09))


# Building a trigram language model with add-λ smoothing example: λ = 0.1.
# 

# In[124]:


#perplexity with lambda=0.1
def perplexity_add_k_smoothing(k,sentence):
  probs=0.0
  count=0
  max_likelyhood=0.0
  for i in range(len(sentence)-2):
    token= sentence[i:i+3]
    #for k in trigram_dict.values():
    max_likelyhood=(trigram_dict.get(token,0.0) + k) / (bigram_dict.get(token[:-1],0.0)+ k*len(trigram_dict))
    max_likelyhood=math.log(max_likelyhood)#log of likelyhood after interpolating linearly with 4 lambdas lamb
    probs=probs + max_likelyhood
    count+=1
  perplexity=2**(-1*probs/count)
  return perplexity  


# In[130]:


test_file_perp={}
for i in range(0,len(test_data)):
  Perp_val=perplexity_add_k_smoothing(0.1,test_data[i])
  test_file_perp[test_files[i]]=Perp_val
  #print(test_files[i]," : ", Perp_val)
sorted_perp_test_files = dict( sorted(test_file_perp.items(),
                           key=lambda item: item[1],
                           reverse=False))
#print('Sorted Dictionary: ')
print(sorted_perp_test_files)


# In[131]:


print(accuracy(sorted_perp_test_files,11.70))


# 

# I found from above code and experiment that add lambda/k smoothing easier to implement and gives better results i.e. higher accuracy as compared to linear interpolation.
# 
# However theoretically and from intuition, interpolation should provide much better results as it considers better range of n-gram probabilities.
# 
# Interpolation can deal with token not found in vocabulary condition much better than add lambda method as in add lambda method we are assigning very low value to the missing token.
# 
# [Your text here.]
