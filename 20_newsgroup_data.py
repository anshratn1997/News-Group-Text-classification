
# coding: utf-8

# In[198]:


from __future__ import print_function
import os
from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))
import logging
import numpy as np
import pandas as pd
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import Counter
import string
from sklearn.model_selection import train_test_split


# In[186]:


# this funtion is to fetch all file name and put in to a list x 
# y value is corresponding to newgroup which vary from 0 to 19 to represent all 20 newsgroup
def get_data(path):
    x=[]
    y=[]
    i=0
    for newspaper in os.listdir(path):
        n_path=path+'/'+newspaper
        for filename in os.listdir(n_path):
            new_path=n_path+'/'+filename
            x.append(new_path)
            y.append(i)
        i+=1
    return x,y


# In[164]:


# this function is to remove punctuation and then tokenize it 
def get_tokens(doc):
        deletechars={ord(c): None for c in string.punctuation} # to remove punctuation
        no_punctuation=doc.translate(deletechars)
        tokens=nltk.word_tokenize(no_punctuation)
        return tokens


# In[180]:


# this funtion is to tokenize all document into a token_data and also put tokens for every 
# individual newsgroup into a seperate list 
def tokenization(x_train,y_train):
    token_data=[]
    token_list=[[] for i in range(20)]
    for i in range(len(x_train)):
        file=open(x_train[i],'r')
        doc=file.read()
        lowers=doc.lower()
        token=get_tokens(lowers)
        filter_token=[w for w in token if not w in stopset]
        token_data.extend(filter_token)
        token_list[y_train[i]].extend(filter_token)
    return token_data,token_list


# In[167]:


# this funtion is to create an 2d np array of 20 * 20001 to store frequency of feature words for individual newsgroup
def get_dictionary(token_list,feature_name):
    x=np.zeros((20,2000),dtype=int)
    for i in range(20):
        token_data=token_list[i]
        for word in token_data:
            if word in feature_name:
                j=feature_name.index(word)
                x[i][j]+=1
    total=[]
    for i in range(20):
        sum=0
        for j in range(2000):
            sum+=x[i][j]
        total.append(sum)
    t_arr=np.array(total)
    t_arr=t_arr.reshape(len(t_arr),1) 
    x=np.append(x,t_arr,axis=1)    # to add a column in array which store sum of all words for every individual newspaper group
    return x


# In[190]:


# this function is to select top 2000 words from dictionary
def get_feature_name(count):
    feature_name=[word for (word, freq) in count.most_common(2000)]
    return feature_name


# In[170]:


# this function is to calculate probablity of word occurence with lapalce correction
def probablity(dictionary,x,feature_name,current_class):
    output=np.log(1)-np.log(20)
    for word in x:
        if word in feature_name:
            index=feature_name.index(word)
            temp=np.log(dictionary[current_class][index]+1)-np.log(dictionary[current_class][dictionary.shape[1]-1]+2000)
            output+=temp
    return output


# In[171]:


# this is to tokenize word for a file 
def tokenize_word(path):
    file=open(path,'r')
    doc=file.read()
    lowers=doc.lower()
    token=get_tokens(lowers)
    filter_token=[w for w in token if not w in stopset]
    return filter_token


# In[195]:


# this function is to provide y predicted value
def predict(dictionary,feature_name,x_test):
    y_pred=[]
    for i in range(len(x_test)):
        words=tokenize_word(x_test[i])
        best_p=-1e9
        best_class=-1
        for j in range(20):
            current=probablity(dictionary,words,feature_name,j)
            if(current>best_p):
                best_p=current
                best_class=j 
        y_pred.append(best_class)
    return y_pred


# In[ ]:


path='../datasets/20_newsgroups'
x,y=get_data(path)


# In[187]:


x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,test_size=0.25,random_state=42)


# In[ ]:


token_data,token_list=tokenization(x_train,y_train)


# In[ ]:


count=Counter(token_data)


# In[191]:


feature_name=get_feature_name(count)


# In[193]:


dictionary=get_dictionary(token_list,feature_name)


# In[196]:


y_pred=predict(dictionary,feature_name,x_test)


# In[197]:


from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

