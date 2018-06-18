#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:15:29 2018

@author: mikawang
"""

import nltk
import re
import numpy as np
import pandas as pd

path_hmg = "/Users/mikawang/works/Health Informatics/summer2018/exercises/hmg.txt"
raw_hmg = open(path_hmg)
path_stopwords = "/Users/mikawang/works/Health Informatics/summer2018/exercises/stopwords.txt"
raw_stopwords = open(path_stopwords)

hmg = raw_hmg.read()
hmg_list = re.split('\n',hmg)
stopwords = raw_stopwords.read()
stopwords_list = re.split('\n',stopwords)

hmg_separated = []
hmg_data = []
hmg_tags = []
hmg_PMID = []

for item in hmg_list:
    separated = re.split('\t',item)
    if len(separated) != 1:
        hmg_separated = hmg_separated + [separated]


for item in hmg_separated:    
    hmg_data = hmg_data + [item[2]+' '+ item[3]] #title and abstract for each data
    hmg_tags = hmg_tags + [item[1]]#label for each data
    hmg_PMID = hmg_PMID + [item[0]]#PMID for each data





tfidf_hmg_VCGS = get_tfidf_VCGS(hmg_data, 0.05, 5)#get_tfidf_VCGS is a function which is the same as I implemented on the previous inspec datas. Here the R is set to 5 while D is set to 5 percent
tfidf_hmg_VCGS_1 = tfidf_hmg_VCGS.values.tolist()
clusters, clusters_dict, clusters_docs = maximum_clustering(tfidf_hmg_VCGS_1, 0.9)#also same as I used before
clusters_Auto_dict = clusters_dict
clusters_Auto = clusters_docs

#clusters for Mesh
hmg_tags_set = list(set(hmg_tags))
clusters_Mesh = [[]] * len(hmg_tags_set)
for item in hmg_separated:
    item_tag = item[1]
    item_index = hmg_separated.index(item)
    cluster_index = hmg_tags_set.index(item_tag)
    clusters_Mesh[cluster_index] = clusters_Mesh[cluster_index] + [item_index]

#how many documents in the Mesh clusters match the ones in each auto cluster
Auto_homo_check = [[]] * len(clusters_Auto)
Auto_centroids = list(clusters_Auto_dict.keys())

for i in range(len(clusters_Auto)):
    auto_centroids = clusters_Auto[i]
    for docs in clusters_Mesh:
        intersection = len(set(auto_centroids).intersection(docs))
        Auto_homo_check[i] = Auto_homo_check[i] + [intersection]

#number of documents in each Mesh label
number_Mesh = []
for docs in clusters_Mesh:
    number_of_docs = len(docs)
    number_Mesh = number_Mesh + [number_of_docs]

#homogenety
homo_for_auto = []

for cluster in Auto_homo_check:
    index_Mesh = sorted(range(len(cluster)), key=lambda i: cluster[i])[-1:]
    tag = hmg_tags_set[index_Mesh[0]]
    if sum(cluster) != 0:
        homo = cluster[index_Mesh[0]]/sum(cluster)
    else:
        homo = 0
    homo_tuple = (homo, tag)    
    homo_for_auto = homo_for_auto + [homo_tuple]
    















