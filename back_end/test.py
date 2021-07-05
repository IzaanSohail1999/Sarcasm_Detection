from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import nltk
import json
import math
from nltk.util import pr
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
import os
from joblib import dump, load
Size = 0


def cleaner(uncleaned):
    porter = PorterStemmer()
    lemma = WordNetLemmatizer()
    uncleaned = uncleaned.lower()
    uncleaned = uncleaned.strip()
    uncleaned = uncleaned.translate(
        {ord(i): None for i in '!\\@#-_:$%^&*();.,?/1”2’3“4‘567890\'\"'})
    for i in uncleaned:
        t = ord(i)
        if t < 97 or t > 122:
            uncleaned = uncleaned.replace(i, "")
    uncleaned = porter.stem(uncleaned)
    uncleaned = lemma.lemmatize(uncleaned, pos='v')
    return uncleaned



Raw_Data = dict()
count = 0
i=0
for lines in open('Sarcasm_Headlines_Dataset.json', 'r'):
    if count == 100:
        break
    count +=1
    Raw_Data[i] = json.loads(lines)
    query = Raw_Data[i]["headline"]
    result = Raw_Data[i]["is_sarcastic"]

    
    with open('IDF_Dictionary.json','r') as json_file:
        idf2 = json.load(json_file)
    vocab_len = len(idf2)
    query_dataset = np.empty((1, vocab_len))
    j = 0
    for i in query_dataset:
        for x in i:
            # print(x)
            query_dataset[0][j] = 0
            j+=1

    tf_query = idf2.copy()
    for key in tf_query.keys():
        tf_query[key] = 0.0
    words = word_tokenize(query)

    stop_word = []
    with open("Stopword-List.txt", 'r') as stop:
        for line in stop:
            temp = line.strip()
            stop_word.append(temp)

    for word in words:
        if len(word) >= 3:
            if word in stop_word:
                continue
            taggs = nltk.tag.pos_tag([word])
            if(taggs[0][1] != 'NNP' and taggs[0][1] != 'FW' and taggs[0][1] != 'PRP'):
                word = cleaner(word)
                if len(word) >= 3:                 
                    # Tokens.append(word)
                    
                    if word in tf_query.keys():
                        if tf_query[word] == 0:
                            tf_query[word] = idf2[word]
                        else:
                            tf_query[word] += idf2[word]
        
    # print(lexicon)
    

    j = 0
    for i in tf_query.values():
        query_dataset[0][j] = i
        j+=1

    # print(query_dataset.shape)

    pca_q = load('pca.joblib') 

    # query_dataset.shape

    # print("Data Processing Completed........")
    # print("")
    # print("Applying PCA.....................")

    # test_pca, scaled_data = applyPCA(query_dataset)
    # print("In function")
    # scaler2 = StandardScaler()

    scaler2 = load('scaler.joblib')

    #scaler=MinMaxScaler()
    # print("Standard scaler defined")
    # scaler2.fit(query_dataset)
    # print("scaler Fit defined")
    scaled_data_query = scaler2.transform(query_dataset)
    # print("scaler trasnform")

    # pca_q = PCA(n_components=3000)
    # pca_q.fit(scaled_data_query)
    test_pca = pca_q.transform(scaled_data_query)
    # print("Data Variance: ", sum(pca_q.explained_variance_ratio_)*100)

    # print(test_pca,scaled_data_query)
    # print("")
    # print("PCA Applied......................")
    # print("")
    # print("Traning Model....................")

    # print( test_pca.shape)
    # print(test_pca)
    new_model = load('model_70acc.joblib') 

    print("predicted resutl",new_model.predict(test_pca),"original result: ",result,"headline:",query)

    i+=1