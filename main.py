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
from function import Read_Data_From_File,Read_Data_From_File1,make_idf, tf_idf_lexicon, applyPCA, sort

def main():
   lexicon = Read_Data_From_File()
   # print(lexicon)
   # idf = make_idf(lexicon)
   # print("Idf made")
   # tf_idf = tf_idf_lexicon(lexicon,idf)
   # print("tf_Idf made")
   # # tf_idf = sort(tf_idf)
   # with open("TFIDF_Dictionary.json", "w") as f: # Writing the index to JSON File.
   #    f.write(json.dumps(tf_idf, sort_keys=False, indent=4)) 
   # with open('TFIDF_Dictionary.json') as json_file:
   #    Dict = json.load(json_file)
   # Dict = sort(Dict)
   # print(Dict)
   # with open("TFIDF_Dictionary_new.json", "w") as f:
      # f.write(json.dumps(Dict, sort_keys=False, indent=4))
   
   #Create Panda DataFrame
   # df = pd.DataFrame(tf_idf)
   # df = np.transpose(df)
   # print(df)

   # print("Data Processing Completed........")
   # print("")
   # print("Applying PCA.....................")

   # x_pca, scaled_data = applyPCA(df)
   
   # print(x_pca,scaled_data)
   # print("")
   # print("PCA Applied......................")
   # print("")
   # print("Traning Model....................")
if __name__ == "__main__":
    main()
