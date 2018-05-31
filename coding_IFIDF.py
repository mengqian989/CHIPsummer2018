#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:57:45 2018

@author: mikawang
"""

import nltk
import re

path_data = "/Users/mikawang/works/Health Informatics/summer2018/inspec.csv"
raw_data = open(path_data)
path_stopwords = "/Users/mikawang/works/Health Informatics/summer2018/stopwords.txt"
raw_stopwords = open(path_stopwords)

inspec = raw_data.read()
inspec_list = re.split('\n',inspec)
stopwords = raw_stopwords.read()
stopwords_list = re.split('\n',stopwords)


#REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
#BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

def text_prepare(text):
    text = text.lower()# lowercase text
    #text = re.sub(REPLACE_BY_SPACE_RE, " ", text)# replace REPLACE_BY_SPACE_RE symbols by space in text
    #text = re.sub(BAD_SYMBOLS_RE, "", text)# delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.split(" ", text)
    text = [item for item in text if item not in stopwords_list and item != '']# delete stopwords from text
    text = [item for item in text if item.isalnum()]
    return text

tokens = []
for text in inspec_list:
    text_1 = text_prepare(text)
    tokens = tokens + [text_1]

#tokens = []
#for text in text_processed:
#    text_2 = re.split(' ', text)
#    tokens = tokens + [text_2]
    
tokens_all = []
for tokens_1 in tokens:
    tokens_all = tokens_all + tokens_1

types_all = sorted(set(tokens_all))
indexed_types = list(enumerate(list(types_all)))

#IDF
doc_size = len(inspec_list)
term_size = len(types_all)

#document frequency
df = []
for item in types_all:
    n = 0
    for tokens_2 in tokens:
        for token in set(tokens_2):
            if item == token:
                n = n + 1
    df = df + [n]

df_pairs = []                
for i in range(term_size):
    pair = (types_all[i],df[i])  
    df_pairs = df_pairs + [pair]

df_pairs_1 = []
for pair in df_pairs:
    if pair[1] >= doc_size * 0.02:
        df_pairs_1 = df_pairs_1 + [pair]

df_frequences= [freq for (word, freq) in df_pairs_1]
df_words= [word for (word, freq) in df_pairs_1]

idf = []
for num in df_frequences:
    idf_1 = np.log(doc_size/num) + 1
    #idf_1 = float(round(idf_1,2))
    idf = idf + [idf_1]

term_size_processed = len(df_words)

#tf = np.zeros([doc_size,term_size_processed])
df_words_indexed = list(enumerate(df_words)) 

def generate_tf(tokens,doc_size,df_words_indexed):
    term_size = len(df_words_indexed)
    tf = np.zeros([doc_size,term_size])
    n = 0
    for tokens_3 in tokens:
        result_vector = np.zeros(term_size)
        for token in tokens_3:
            if token in df_words:
                index = [index for (index,word) in df_words_indexed if word == token]
                if index != []:
                    index = index[0]
                    result_vector[index] = result_vector[index] + 1
        #result_vector = np.multiply(result_vector, (1/len(tokens_3)))            
        tf[n] = result_vector
        n = n + 1
    return tf

tf_inspec = generate_tf(tokens,doc_size,df_words_indexed)

tfidf = np.zeros([doc_size,term_size_processed])
n = 0
for item in tf_inspec:
    new_array = np.multiply(item,idf)
    tfidf[n] = new_array
    n = n + 1

#normalize the tf-idf value
tfidf_norm = []
for vec in tfidf:
    dot = np.dot(vec,vec)
    norm = np.multiply(vec, (1/np.sqrt(dot)))
    norm = list(norm)
    tfidf_norm = tfidf_norm + [norm]

tfidf_title = [df_words]+ tfidf_norm




import pandas as pd
tfidf_df_2 = pd.DataFrame(tfidf_title[1:])
tfidf_df_2.columns = tfidf_title[0]
tfidf_df_2.to_csv('tfidf_Mika4.csv')



