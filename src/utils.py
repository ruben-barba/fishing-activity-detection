import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def min_max(df,col):
    '''
    function that applies a min max standarization
    '''
    max_val = df[col].max()
    min_val = df[col].min()
    min_maxed = (df[col] - min_val) / (max_val-min_val)   
    return min_maxed

def check_min_max(df,to_check_col,checker_col):
    '''
    function that returns 1 if the target col is min maxed, else 0
    '''
    # first create a col normalized with the min max method
    check = min_max(df,to_check_col)
    # now compare the target col and the normalized col
    return (df[checker_col] == check).mean() 

def check_categorical(df):
    '''
    function that takes a df and returns a pandas serie with the unique vales of each feature
    '''
    uniques = [len(df[i].unique()) for i in df.columns]
    dictionary = {k:v for k,v in zip(df.columns,uniques)}
    return pd.Series(dictionary)
