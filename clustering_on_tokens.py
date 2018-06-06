#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:58:21 2018

@author: mikawang
"""
#tokens clustering
tfidf_tokens = tfidf_features.transpose()

#def maximum_clustering_tokens(tfidf_features,theta,features):
dff_words_indexed = list(enumerate(dff_words))
all_distance = []

#get the first centroid
n = 0
for vec in tfidf_tokens:
    if sum(vec) != 0:
        init_centroid = vec
        break
    else:
        n = n + 1

first_centroid_index = n    
init_distance = 0
all_centroid = [init_centroid]
all_centroid_token = [(first_centroid_index, dff_words[first_centroid_index])]

#get the second centroid
n_2 = 0
for vec in tfidf_tokens:
    if sum(vec) != 0: 
        distance = get_distance(init_centroid,vec)
        if distance > init_distance:
            init_distance = distance
            centroid_2 = vec
            second_centroid_index = n_2
    n_2 = n_2 + 1
    centroid_token = dff_words[second_centroid_index]
all_distance = all_distance + [init_distance]
all_centroid = all_centroid + [centroid_2]
all_centroid_token = all_centroid_token + [(second_centroid_index, centroid_token)]


#get the maximum distance

init_distance_max = init_distance

while init_distance_max > theta * np.mean(all_distance):
    init_distance_max = 0
    n_3 = 0
    for vec in tfidf_tokens:
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
                new_centroid_index = n_3
                new_centroid_token = dff_words[new_centroid_index]
        n_3 = n_3 + 1
    if init_distance_max > AP * np.mean(all_distance):
        all_distance = all_distance + [init_distance_max]
        all_centroid = all_centroid + [new_centroid]
        all_centroid_token = all_centroid_token + [(new_centroid_index, new_centroid_token)]
        

    
#clustering    
centroid_indexed = list(enumerate(all_centroid))
clusters = [[]] * (len(all_centroid)+1)
clusters_tokens = [[]] * (len(all_centroid)+1)

init_distance_min = 1
n = 0
for vec in tfidf_tokens:
    init_distance_min = 1
    if sum(vec) !=0:
        for cen_pair in centroid_indexed:
            index = cen_pair[0]
            centroid = cen_pair[1]
            distance = get_distance(centroid,vec)
            if distance <= init_distance_min:
                init_distance_min = distance
                cen_pair_final = cen_pair
                word_index = n
        clus_index = cen_pair_final[0]
        clusters[clus_index] = clusters[clus_index] + [vec]
        clusters_tokens[clus_index] = clusters_tokens[clus_index] + [dff_words[word_index]]
    else:
        clusters[-1] = clusters[-1] + [vec]
    n = n + 1


#make a dictionary for the token clusters
centroid_key = [centroid for (index,centroid) in all_centroid_token]

cluster_token_dict = []
for i in range(len(centroid_key)):
    cluster_token_dict = cluster_token_dict + [(centroid_key[i], clusters_tokens[i])]
cluster_token_dict = dict(cluster_token_dict)
   

# store the result in other variables 
cluster_token_dict_1 = cluster_token_dict
clusters_1 = clusters




    