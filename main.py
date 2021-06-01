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
import os
from function import Read_Data_From_File,make_idf,tf_idf_lexicon

def main():
   lexicon = Read_Data_From_File()
   # print(lexicon)
   print(len(lexicon))
   # idf = make_idf(lexicon)
   # print(len(idf))
   # print(idf)
   # tf_idf = tf_idf_lexicon(lexicon,idf)
   # print(tf_idf)
   # print(len(tf_idf))
   # print(tf_idf.keys())
    
        
if __name__ == "__main__":
    main()
