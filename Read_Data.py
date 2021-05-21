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
Size = 26709
#Classification problem , either sarcastic or not sarcastic ,2 class problem

def Read_Data_From_File():
    alpha = ''
    Raw_Data = dict()
    X_headline =[]
    Y_Sarcastic=[]
    i=0
    #Read data in X and Y labels ,headline and sarcacism score.
    for lines in open('Sarcasm_Headlines_Dataset.json','r'):
        Raw_Data[i]=json.loads(lines)
        X_headline.append(Raw_Data[i]["headline"])
        Y_Sarcastic.append(Raw_Data[i]["is_sarcastic"])
        i=i+1

    #Tokenize
    Tokens = []
    count = 0
    for y in range(len(X_headline)):
        for word in X_headline[y]:
            if(word == ' '):
                count += 1
                Tokens.append(alpha)
                alpha = ''
            else:
                alpha = alpha + word
    print(Tokens)
    print("The word count is: "+str(len(Tokens)))
    print("Number of docs are: " + str(i))
