#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 12:28:14 2018

@author: mikawang
"""

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
        
R = 1
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
len(df_words)

#sorting the tuples which have the same term
mat_on_term = [[]] * len(df_words)
i = 0
for term in df_words:
    for tuples in tuples_above_R:
        for tuple_2 in tuples:
            if term == tuple_2[1]:
                mat_on_term[i] = mat_on_term[i] + [tuple_2]
    i = i + 1

useful_words = []
for item in tuples_above_R:
    for sub_item in item:
        useful_words = useful_words + [sub_item[1]]

# get the terms which show up in more than 5% of all documents                
D = 0.05
mat_on_term_VCGS = []
for item in mat_on_term:
    if len(item) >= 0.05 * doc_size:
        mat_on_term_VCGS = mat_on_term_VCGS + [item]

len([word for (word,freq) in df_pairs_1 if freq >= 0.05*doc_size])      
terms_above_D = [word for (word,freq) in df_pairs_1 if freq >= 0.05*doc_size]
[freq for (word,freq) in df_pairs_1 if word == 'query']


