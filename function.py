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
Size = 0

def make_idf(lexicon):
    idf = dict()
    global Size
    N = Size
    for term,docs in lexicon.items():
        df = len(docs)
        temp = math.log(N / df)
        idf[term] = temp
    return idf

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

def Read_Data_From_File():
    Tokens = []
    alpha = ''
    Raw_Data = dict()
    X_headline = []
    Y_Sarcastic = []
    i = 0
    count = 0
    lexicon = {}

    stop_word = []
    with open("Stopword-List.txt", 'r') as stop:
        for line in stop:
            temp = line.strip()
            stop_word.append(temp)

    for lines in open('Sarcasm_Headlines_Dataset.json', 'r'):
        count += 1
        Raw_Data[i] = json.loads(lines)
        X_headline.append(Raw_Data[i]["headline"])
        Y_Sarcastic.append(Raw_Data[i]["is_sarcastic"])

        #tokenize here
        words = word_tokenize(X_headline[i])
        curr = str(i)
        for word in words:
            if len(word) >= 3:
                if word in stop_word:
                    continue
                taggs = nltk.tag.pos_tag(word)
                if(taggs[0][1] != 'NNP' and taggs[0][1] != 'FW' and taggs[0][1] != 'PRP'):
                    word = cleaner(word)
                    if len(word) >= 3:                 
                        # Tokens.append(word)

                        if word in lexicon:
                            if curr in lexicon[word]:
                                lexicon[word][i] += 1
                            else:
                                lexicon[word][i] = 1
                        else:
                            lexicon[word] = {}
                            if curr in lexicon[word]:
                                lexicon[word][i] += 1
                            else:
                                lexicon[word][i] = 1
        i = i+1
        print(lexicon)
    global Size 
    Size = count
    return lexicon

def tf_idf_lexicon(lexicon, idf):
    tf_idf = {}
    for term, docs in lexicon.items():
        tf_idf[term] = {}
        for docNo, tf in docs.items():
            tf_idf[term][docNo] = tf * idf[term]

    return tf_idf

def applyPCA(df):
    print("In function")
    scaler = StandardScaler()
    #scaler=MinMaxScaler()
    print("Standard scaler defined")
    scaler.fit(df)
    print("scaler Fit defined")
    scaled_data = scaler.transform(df)
    print("scaler trasnform")

    pca = PCA(n_components=900)
    pca.fit(scaled_data)
    x_pca = pca.transform(scaled_data)
    print("Data Variance: ", sum(pca.explained_variance_ratio_)*100)

    return x_pca, scaled_data

def Read_Data_From_File1():
    global Size
    Tokens = []
    alpha = ''
    Raw_Data = dict()
    X_headline = []
    Y_Sarcastic = []
    i = 0
    count = 0
    lexicon = {}

    stop_word = []
    with open("Stopword-List.txt", 'r') as stop:
        for line in stop:
            temp = line.strip()
            stop_word.append(temp)

    for lines in open('Sarcasm_Headlines_Dataset.json', 'r'):
        if count == 2:
            break
        count += 1
        Raw_Data[i] = json.loads(lines)
        X_headline.append(Raw_Data[i]["headline"])
        Y_Sarcastic.append(Raw_Data[i]["is_sarcastic"])

        #tokenize here
        words = word_tokenize(X_headline[i])
        curr = str(i)
        for word in words:
            if len(word) >= 3:
                if word in stop_word:
                    continue
                taggs = nltk.tag.pos_tag(word)
                if(taggs[0][1] != 'NNP' and taggs[0][1] != 'FW' and taggs[0][1] != 'PRP'):
                    word = cleaner(word)
                    if len(word) >= 3:
                        # print(i,word)
                        Tokens.append(tuple((i,word)))

                        # if word in lexicon:
                        #     if curr in lexicon[i]:
                        #         lexicon[i][word] += 1
                        #     else:
                        #         lexicon[i][word] = 1
                        # else:
                        #     lexicon[i] = {}
                        #     if curr in lexicon[i]:
                        #         lexicon[i][word] += 1
                        #     else:
                        #         lexicon[i][word] = 1
        i = i+1
    for i in range(2):
        for x in Tokens:
            if i in lexicon:
                if x[1] in lexicon[i]:
                    if i == x[0]:
                        lexicon[i][x[1]] += 1
                    else:
                        lexicon[i][x[1]] = 0
                else:
                    if i == x[0]:
                        lexicon[i][x[1]] = 1
                    else:
                        lexicon[i][x[1]] = 0
            else:
                lexicon[i] = {}
                if x[1] in lexicon[i]:
                    if i == x[0]:
                        lexicon[i][x[1]] += 1
                    else:
                        lexicon[i][x[1]] = 0
                else:
                    if i == x[0]:
                        lexicon[i][x[1]] = 1
                    else:
                        lexicon[i][x[1]] = 0

    Size = count
    return lexicon,Tokens

def sort(tfidf):
    size = 26709
    list1 = []
    check = False
    for i in range(size):
        list1.append(i+1)
    # print(list1)
    
    for word,doc in tfidf.items():
        print(word)
        for i in list1:
            if i not in doc.keys():
                tfidf[word][i] = 0
    return tfidf

def make_idf1(Tokens):
    list1 = []
    idf = dict()
    global Size
    N = Size
    # for term,docs in lexicon.items():
    #     df = len(docs)
    #     temp = math.log(N / df)
    #     idf[term] = temp
    # return idf
    for i in Tokens:
        # print(i[0],i[1])        
        count =  i[0]
        while i[0]+1 == count:
             list1.append(Tokens[i[1]])
    print(list1)