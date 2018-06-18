#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:44:04 2018

@author: mikawang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:17:40 2018

@author: mikawang
"""

import nltk
import re
import numpy as np
import pandas as pd

def text_prepare(text):
    text = text.lower()# lowercase text
    #text = re.sub(REPLACE_BY_SPACE_RE, " ", text)# replace REPLACE_BY_SPACE_RE symbols by space in text
    #text = re.sub(BAD_SYMBOLS_RE, "", text)# delete symbols which are in BAD_SYMBOLS_RE from text
    text = re.split(r"\W+", text)
    text = [item for item in text if item not in stopwords_list and item != '']# delete stopwords from text
    text = [item for item in text if item.isalnum()]
    return text

def generate_tf(tokens,df_words):# tokens: a list of tokens for all the documents/ df_words: features
    term_size = len(df_words)
    doc_size = len(tokens)
    df_words_indexed = list(enumerate(df_words))
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


def get_tfidf_VCGS(list_of_text, D, R):
    tokens = []
    for text in list_of_text:
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
    #indexed_types = list(enumerate(list(types_all)))
    
    #IDF
    doc_size = len(list_of_text)
    term_size = len(types_all)
    
    #document frequency
    df_1 = np.zeros(len(types_all))
    for tokens_2 in tokens:
        types_2 = set(tokens_2)
        for token in types_2:
            if token in types_all:
                index = types_all.index(token)
                df_1[index] += 1 

    
    df_pairs = []                
    for i in range(term_size):
        pair = (types_all[i],df_1[i])  
        df_pairs = df_pairs + [pair]
    
    df_pairs_1 = []
    for pair in df_pairs:
        if pair[1] >= doc_size * 0.02:
           df_pairs_1 = df_pairs_1 + [pair]
    
    df_frequences= [freq for (word, freq) in df_pairs_1]
    #df_words= [word for (word, freq) in df_pairs_1]
    df_words = [word for (word,freq) in df_pairs_1]
    
    
    idf = []
    for num in df_frequences:
        idf_1 = np.log(doc_size/num+1)
        #idf_1 = float(round(idf_1,2))
        idf = idf + [idf_1]
    
    term_size_processed = len(df_words)
    
    #tf = np.zeros([doc_size,term_size_processed])
    #df_words_indexed = list(enumerate(df_words)) 
    
    tf = generate_tf(tokens,df_words)
    
    tfidf = np.zeros([doc_size,term_size_processed])
    
    n = 0
    for item in tf:
        new_array = np.multiply(item,idf)
        tfidf[n] = new_array
        n = n + 1
    
    #normalize the tf-idf value
    tfidf_norm = []###
    for vec in tfidf:
        dot = np.dot(vec,vec)
        if dot != 0:
            norm = np.multiply(vec, (1/np.sqrt(dot)))
            norm = list(norm)
        else:
            norm = vec   
        tfidf_norm = tfidf_norm + [list(norm)]
    
    tfidf_title = [df_words]+ tfidf_norm
    
    tfidf_dataframe = pd.DataFrame(tfidf_title[1:])
    tfidf_dataframe.columns = tfidf_title[0]
    
    #making the tuples
    ranking_table = []
    i = 0
    for item in tfidf_norm:#normailized tf-idf values
        new_list = []
        for n in range(len(item)):
            value_tfidf = item[n]
            feature_term = df_words[n]
            document_identifier = i
            new_tuple = (document_identifier, feature_term, value_tfidf)
            new_list = new_list + [new_tuple]
        i = i+1
        new_list = sorted(new_list, key = lambda tup: tup[2], reverse = True)
        new_list_1 = []
        for tup_num in range(len(new_list)):
            ranking = tup_num
            new_tuple_1 = new_list[tup_num] + (ranking,)
            new_list_1 = new_list_1 + [new_tuple_1]
        ranking_table = ranking_table + [new_list_1]
            
    tuples_above_R = []
    for list_of_tuples in ranking_table:
        new_tuples = list_of_tuples[:R]
        if new_tuples[-1][2] != 0:
            check_ranking = R+1
            while check_ranking < len(ranking_table):
                check_tuple = list_of_tuples[check_ranking]
                if check_tuple[2] == new_tuples[-1][2]:
                    new_tuples = new_tuples + [check_tuple]
                    check_ranking = check_ranking + 1
                else:
                    break
        else:
            new_tuples_1 = []
            for tuple_1 in new_tuples:
                if tuple_1[2] > 0:
                    new_tuples_1 = new_tuples_1 + [tuple_1]
                    new_tuples = new_tuples_1
        
        tuples_above_R = tuples_above_R + [new_tuples] 
                
    #df_words: The terms used as features, 430 intotal getting rid of low frequency words
    #len(df_words)
    
    #sorting the tuples which have the same term
    mat_on_term = [[]] * len(df_words)
    i = 0
    for term in df_words:
        for tuples in tuples_above_R:
            for tuple_2 in tuples:
                if term == tuple_2[1]:
                    mat_on_term[i] = mat_on_term[i] + [tuple_2]
        i = i + 1
            
    terms_above_D = [word for (word,freq) in df_pairs_1 if freq >= D * doc_size]
    
    features = []
    for word in terms_above_D:
        if word in df_words:
            features = features + [word]
    
    tfidf_dataframe = tfidf_dataframe[features]
    #tfidf_features = tfidf_dataframe.values.tolist()
            
    #df_words_index = list(enumerate(df_words))
    #df_words_index_1 = []
    #for tuple_1 in df_words_index:
    #    index = tuple_1[0]
    #    word = tuple_1[1]
    #    df_words_index_1 = df_words_index_1 + [(word,index)]
     
    #df_words_dict = dict(df_words_index_1)
    
    #indexes = [df_words_dict[term] for term in features]
    
    
    
    #tfidf_title = [df_words]+ tfidf_norm
    return tfidf_dataframe

tfidf_hmg_VCGS_1 = tfidf_hmg_VCGS.value.tolist()

tfidf = tfidf_hmg_VCGS_1






import pandas as pd
tfidf_df_2 = pd.DataFrame(tfidf_title[1:])
tfidf_df_2.columns = tfidf_title[0]
#tfidf_df_2.to_csv('tfidf_Mika4.csv')

tfidf_df_centroid = tfidf_df_2[features]
tfidf_df_centroid[dates[5]]

n = 0
for vec in tfidf_hmg:
    check = sum(vec)
    if check == 0:
        n = n+1


