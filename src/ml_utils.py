import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler

import random

def metrics(y_test, y_train, y_test_pred, y_train_pred, model_type):
    '''
    return the accuracy, recall and f1-score of both train and test data
    
    :params y_test, y_train, y_test_pred, y_train_pred: the train test split
    :type y_test, y_train, y_test_pred, y_train_pred: dataframe or np.array
    
    :param model_type: the model
    :type model_type: string
    
    :return: a dataframe with scores
    '''
    
    results = {'Accuracy': [accuracy_score(y_test, y_test_pred), accuracy_score(y_train, y_train_pred)],
               'Recall': [recall_score(y_test, y_test_pred), recall_score(y_train, y_train_pred)],
                'F1': [f1_score(y_test, y_test_pred), f1_score(y_train, y_train_pred)],
                 "set": ["test", "train"]}
    df = pd.DataFrame(results)
    df["model"] = model_type
    return df 

def correlation_heatmap(df):
    '''
    given a dataframe returns a triangle heatmap
    
    :param df: the data you want to plot correlations
    :type df: dataframe
    
    :return: a triangle heatmap
    '''
    plt.figure(figsize=(16, 6))
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))
    heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
    heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=15)
    
    return heatmap

def feature_importances_rf(X_train, X_test, y_train, y_test,model=None):
    '''
    function that uses a RandomForestClassifier to find the feature importances
    
    :param X_train, X_test, y_train, y_test: the data you want to find the feature importances
    :type X_train, X_test, y_train, y_test: dataframe or np.array
    
    :return: a bar plot with the sorted feature importances 
    '''
    
    if model == None:  # future work that allows to select the model you want yo use
        model = RandomForestClassifier()

    model.fit(X_train, y_train)
    feature_importances = {k:v*100 for k,v in zip(X_train.columns,model.feature_importances_)}
    feature_importances_df = pd.DataFrame(data=feature_importances.values(),
                                          index=feature_importances.keys(),
                                          columns=['percentage']) 

    df = feature_importances_df.sort_values('percentage',ascending=True) 
    
    plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
    plt.rcParams.update({'font.size': 12})
    
    plot = plt.barh(df.index,df['percentage'])   
    
    return plot

def min_max_scaler(train_col,test_col):
    '''
    feature that fits a MinMaxScaler with the train column, and use the same fit on the test column
    
    :param train_col,test_col: columns you want to min max
    :type train_col,test_col: iterable (pandas DataFrame, pandas Serie, list, np array)
    
    :return: both columns min maxed (2 vars)
    '''
    
    minmax = MinMaxScaler()
    minmax.fit(train_col) 
    train_col_normalized = minmax.transform(train_col)
    
    minmax.fit(test_col) 
    test_col_normalized = minmax.transform(test_col)
    
    return train_col_normalized, test_col_normalized

def set_threshold(df,threshold):
    '''
    feature that set a threshold and applies it to the is_fishing column
    
    :param df: columns you want to min max
    :type df: dataframe
    
    :param threshold: a number [0,1], i.e: threshold==0.6, all values bigger than 0.6 will be set on 1
    :type threshold: float
    
    :return: the column with only 0 and 1 values
    '''
    threshold_col = np.where(df.is_fishing > threshold,1,0)
    
    return threshold_col

def mmsi_split(df,split_size):
    '''
    function that splits the data by mmsi number; this assures not split a single vessel during the train test split, only complete vessels will be splitted
    
    --future work--: Do a while loop to add more mmsi numbers to the list until reach the desired % split
    
    :param df: the data
    :type df: dataframe
    
    :param split_size: a number [0,1], means the size of one chunk
    :type split_size: float
    
    :return: 2 vars, data splitted by mmsi
    '''
    # Chunk the mmsi list
    list_chunk = df['mmsi'].unique()
    random.shuffle(list_chunk)
    chunk_size = int(round(len(list_chunk) * split_size,0))
    
    chunk_1 = list_chunk[:chunk_size]
    chunk_2 = list_chunk[chunk_size:]
    
    split_1 = df['mmsi'].isin(chunk_1)
    split_2 = df['mmsi'].isin(chunk_2)
    
    # Use the chunks to split the data
    data_1 = df[split_1]
    data_2 = df[split_2]
    
    return data_1, data_2
    
def train_test_mmsi_split(X,y,train_size):
    '''
    function that splits the data in train and test by mmsi number; this assures not split a single vessel during the train test split, only complete vessels will be splitted
    
    --next steps--: Do a while loop to add more mmsi number to the list until reach the desired % split
    
    :param df: the data
    :type df: dataframe
    
    :param split_size: a number [0,1], means the size of one chunk
    :type split_size: float
    
    :return: 4 vars, data splitted by mmsi
    '''
    # Chunk the mmsi list
    list_chunk = X['mmsi'].unique()
    random.shuffle(list_chunk)
    chunk_size = int(round(len(list_chunk) * train_size,0))
    train_mmsi = list_chunk[:chunk_size]
    test_mmsi = list_chunk[chunk_size:]
    
    train_split = X['mmsi'].isin(train_mmsi)
    test_split = X['mmsi'].isin(test_mmsi)
    
    # Use the chunks to split the data
    X_train = X[train_split]
    X_test = X[test_split]
    
    y_train = y[train_split]
    y_test = y[test_split]
    
    print(f'Train: {100-round(X_test.shape[0]/X_train.shape[0]*100,1)}% | Test: {round(X_test.shape[0]/X_train.shape[0]*100,1)}%' )
    
    return X_train, X_test, y_train, y_test

def train_test_val_mmsi_split(X,y,train_size,val_size):
    '''
    function that splits the data in train, test and validation by mmsi number; this assures not split a single vessel during the train test split, only complete vessels will be splitted
    
    --next steps--: Do a while loop to add more mmsi number to the list until reach the desired % split
    --next steps--: Improve this function!
    
    :param df: the data
    :type df: dataframe
    
    :param split_size: a number [0,1], means the size of one chunk
    :type split_size: float
    
    :return: 6 vars, data splitted by mmsi
    '''
    
    test_size = 1-train_size-val_size

    # Chunk the mmsi list
    list_chunk = X['mmsi'].unique()
    random.shuffle(list_chunk)

    chunk_train = int(round(len(list_chunk) * train_size,0))
    chunk_val = int(round(len(list_chunk) * val_size,0))

    val_mmsi = list_chunk[:chunk_val]
    train_mmsi = list_chunk[chunk_val:chunk_train]
    test_mmsi = list_chunk[chunk_train:]

    train_split = X['mmsi'].isin(train_mmsi)
    test_split = X['mmsi'].isin(test_mmsi)
    val_split = X['mmsi'].isin(val_mmsi)

    # Use the chunks to split the data
    X_train = X[train_split]
    X_test = X[test_split]
    X_val = X[val_split]

    y_train = y[train_split]
    y_test = y[test_split]
    y_val = y[val_split]

    print(f'Train: {round(X_train.shape[0]/X.shape[0]*100,1)}% |\
          Test: {round(X_test.shape[0]/X.shape[0]*100,1)}% |\
          Validation: {round(100-round(X_train.shape[0]/X.shape[0]*100,1)-round(X_test.shape[0]/X.shape[0]*100,1),1)}%')
    
    return X_train, X_test, X_val, y_train, y_test, y_val