#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 12:38:22 2018

@author: mikawang
"""

#the 110 terms which used as features were extracted with R = 5 and D = 0.5

#get the frequency for the 110 features
df_pairs_features = []
for item in df_pairs_1: #df_pairs_1 is a list consists of tuples of terms and document frequency
    if item[0] in features: #features is a list consists of the 110 terms 
        df_pairs_features = df_pairs_features + [item]

#seperate the frequency and words for the 110 features
dff_frequences = [freq for (word, freq) in df_pairs_features]
dff_words = [word for (word, freq) in df_pairs_features]

#idf for 110 features
idf_features = []
for num in dff_frequences:
    idf_2 = np.log(doc_size/num+1)
    #idf_1 = float(round(idf_1,2))
    idf_features = idf_features + [idf_2]

#features index
dff_words_indexed = list(enumerate(dff_words)) 

def generate_tf(tokens,df_words):
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

#term frequency for features
tf_features = generate_tf(tokens, dff_words)

#tfidf for 110 features

tfidf_features = np.zeros([doc_size,len(dff_words)])

n = 0
for item in tf_features:
    new_array = np.multiply(item,idf_features)
    tfidf_features[n] = new_array
    n = n + 1

#normalize the tf-idf for features
tfidf_features_norm = []
for vec in tfidf_features:
    dot = np.dot(vec,vec)
    norm = np.multiply(vec, (1/np.sqrt(dot)))
    norm = list(norm)
    tfidf_features_norm = tfidf_features_norm + [norm]


def get_distance(vec1, vec2):
    dot_vec1 = np.dot(vec1,vec1)
    dot_vec2 = np.dot(vec2,vec2)
    divider = np.sqrt(dot_vec1 * dot_vec2)
    upper = sum(np.multiply(vec1,vec2))
    substract = upper/divider
    result = 1 - substract
    return result

all_distance = []
#get the first centroid
n = 0
for vec in tfidf_features:
    if sum(vec) != 0:
        init_centroid = vec
        centroid_index = n
        break
    else:
        n = n + 1
    
init_distance = 0
theta = 0.9
all_centroid = [init_centroid]
all_centroid_doc = [centroid_index] #these list contains the original document index

#get the second centroid
n = 0
for vec in tfidf_features:
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
AP = 0.9
while init_distance_max > AP * np.mean(all_distance):
    init_distance_max = 0
    n = 0
    for vec in tfidf_features:
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
    if init_distance_max > AP * np.mean(all_distance):
        all_distance = all_distance + [init_distance_max]
        all_centroid = all_centroid + [new_centroid]
        all_centroid_doc = all_centroid_doc + [new_centroid_index]
        

#clustering
centroid_indexed = list(enumerate(all_centroid)) #these are the new indexes for the centroids
clusters = [[]] * (len(all_centroid)+1)
clusters_doc = [[]] * (len(all_centroid)+1)


init_distance_min = 1
n = 0
for vec in tfidf_features:
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
        clusers_doc[clus_index] = clusers_doc[clus_index] + [doc_index]      
    else:
        clusters[-1] = clusters[-1] + [vec]
        clusters[-1] = clusters[-1] + [doc_index]
    n = n + 1

#make a dictionary for the token clusters
centroid_key = [centroid for (index,centroid) in all_centroid_doc]
cluster_doc_dict = []
for i in range(len(centroid_key)):
    cluster_doc_dict = cluster_doc_dict + [(centroid_key[i], clusters_doc[i])]
cluster_doc_dict = dict(cluster_doc_dict)

# store the result in other variables 
cluster_doc_dict_1 = cluster_doc_dict
clusters_1 = clusters
