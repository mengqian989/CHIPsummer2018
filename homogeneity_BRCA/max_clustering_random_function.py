#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:43:04 2018

@author: mikawang
"""

import nltk
import re
import numpy as np
import pandas as pd
import random


def get_distance(vec1, vec2):
    dot_vec1 = np.dot(vec1,vec1)
    dot_vec2 = np.dot(vec2,vec2)
    divider = np.sqrt(dot_vec1 * dot_vec2)
    upper = sum(np.multiply(vec1,vec2))
    substract = upper/divider
    result = 1 - substract
    return result

def maximum_clustering_random(tfidf, theta):
    #get the first centroid randomly
    init_centroid = 0
    while init_centroid == 0:
        random_index = random.randint(0,(len(tfidf)-1))
        init_centroid_temp = tfidf[random_index]
        if sum(init_centroid_temp) != 0:
            init_centroid = init_centroid_temp
            centroid_index = random_index
    
    all_distance = []    
    init_distance = 0
    all_centroid = [init_centroid]
    all_centroid_doc = [centroid_index] #these list contains the original document index
    
    #get the second centroid
    n = 0
    for vec in tfidf:
        if sum(vec) != 0: 
            distance = get_distance(init_centroid,vec)
            if distance > init_distance:
                init_distance = distance
                centroid_2 = vec
                centroid_index = n
        n = n + 1
    all_distance = all_distance + [init_distance]
    all_centroid = all_centroid + [centroid_2]
    all_centroid_doc = all_centroid_doc + [centroid_index]
    
    init_distance_max = init_distance
    
    #get the maximum distance
    while init_distance_max > theta * np.mean(all_distance):
        init_distance_max = 0
        n = 0
        for vec in tfidf:
            if sum(vec) != 0:
                distance_0 = 1
                for centroid in all_centroid:
                    distance_1 = get_distance(vec, centroid)
                    if distance_1 <= distance_0:
                        distance_0 = distance_1
                        centroid_repeat = vec
                if distance_0 > init_distance_max:
                    init_distance_max = distance_0
                    new_centroid = centroid_repeat
                    new_centroid_index = n
            n = n + 1
        if init_distance_max > theta * np.mean(all_distance):
            all_distance = all_distance + [init_distance_max]
            all_centroid = all_centroid + [new_centroid]
            all_centroid_doc = all_centroid_doc + [new_centroid_index]
            
    
    #clustering
    centroid_indexed = list(enumerate(all_centroid)) #these are the new indexes for the centroids
    clusters = [[]] * (len(all_centroid)+1)
    clusters_doc = [[]] * (len(all_centroid)+1)
    
    
    init_distance_min = 1
    n = 0
    for vec in tfidf:
        init_distance_min = 1
        if sum(vec) !=0:
            for cen_pair in centroid_indexed:
                index = cen_pair[0]
                centroid = cen_pair[1]
                distance = get_distance(centroid,vec)
                if distance <= init_distance_min:
                    init_distance_min = distance
                    cen_pair_final = cen_pair
                    doc_index = n
            clus_index = cen_pair_final[0]
            clusters[clus_index] = clusters[clus_index] + [vec]
            clusters_doc[clus_index] = clusters_doc[clus_index] + [doc_index]      
        else:
            clusters[-1] = clusters[-1] + [vec]
            clusters[-1] = clusters[-1] + [doc_index]
        n = n + 1
    
    #make a dictionary for the token clusters
    centroid_key = all_centroid_doc + ['others']
    cluster_doc_dict = []
    for i in range(len(centroid_key)):
        cluster_doc_dict = cluster_doc_dict + [(centroid_key[i], clusters_doc[i])]
    cluster_doc_dict = dict(cluster_doc_dict)
    
    return clusters, cluster_doc_dict, clusters_doc






