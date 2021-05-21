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
from Read_Data import Read_Data_From_File

def main():
   Read_Data_From_File()
    
        
if __name__ == "__main__":
    main()