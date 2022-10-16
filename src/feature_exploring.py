import pandas as pd
import numpy as np

'''
due to the high number of features of the data, this doc is aimed to explore and select the right features with ease
'''

def get_window_features(df,window):
    '''
    function that selects the features of a given window
    
    :param df: the data
    :type df: dataframe
    
    :param window: the substring of the column you want to find to select them all
    :type window: string
    
    :return: a list with the selected columns
    '''
    cols = np.array(df.columns) # pandas columns to array
    # Building a boolean mask: False to the features not containing the window in his name, True the rest
    mask = np.where(np.char.find(cols.astype(str), window)==-1,False,True)
    # Mask filtering on df.columns
    window_cols = df.columns[mask]
    
    return window_cols

def get_multiple_window_features(df,window_list):
    '''
    function that returns the features given a window list
    
    :param df: the data
    :type df: dataframe
    
    :param window: the substring of the column you want to find to select them all
    :type window: string
    
    :return: a list with the selected columns
    '''
    window_features = [get_window_features(df,window) for window in window_list]
    return np.array(window_features).flatten()

def get_no_window_features(df):
    '''
    given a dataframe, return all the not windowed features
    
    :param df: the data
    :type df: dataframe
    
    :return: a list with no windowed features
    '''
    all_windows = ['1800','3600','10800','21600','43200','86400']
    df = df.drop(get_multiple_window_features(df,all_windows),axis=1)
    return df.columns

def get_all_except_features(df,keyword):
    '''
    given a dataframe, return all the features that not containing the keyword

    :param df: the data
    :type df: dataframe
    
    :param keyword: the word you want to match
    :type keyword: string
    
    :return: a list with the features not containing the keyword
    '''
    cols = np.array(df.columns) # pandas columns to array
    # Building a boolean mask: False to the features not containing the window in his name, True the rest
    mask = np.where(np.char.find(cols.astype(str), keyword)==-1,True,False)
    # Mask filtering on df.columns
    window_cols = df.columns[mask]
    
    return window_cols

def column_select(df,drop_always=None,drop_y=None,drop_mmsi=None,drop_gear=None,col_groups=None):
    '''
    this function selects a combination of columns depending on the args passed.
    
    :param df: the data
    :type df: dataframe
    
    :param drop_always: True if you want to drop the following columns: ['lat','lon','timestamp','speed','course','measure_distance_from_port']
    :type drop_always: boolean
    
    :param drop_y: True if you want to drop the y feature
    :type drop_y: boolean
    
    :param drop_mmsi: True if you want to drop the mmsi feature
    :type drop_mmsi: boolean
    
    :param drop_gear: True if you want to drop the gear_type feature
    :type drop_gear: boolean
    
    :param col_groups: keyword of the group of columns you want to select, i.e: ['1800','3600'] will select all the columns containing these substrings
    :type col_groups: iterable (list,array,tuple)
    
    :return: a list with the features not containing the keyword
    '''
    cols_to_return = []
    
    # drop the non valuable columns for our model, only if the variable is True
    if drop_always:
        df = df.drop(['lat','lon','timestamp','speed','course','measure_distance_from_port'],axis=1)
    if drop_y:
        df = df.drop(['is_fishing'],axis=1)
    if drop_mmsi:
        df = df.drop(['mmsi'],axis=1)
    if drop_gear:
        df = df.drop(['gear_type'],axis=1)
    
    # get the normal cols (no windowed ones)   
    all_windows = ['1800','3600','10800','21600','43200','86400']
    normal_cols = get_multiple_window_features(df,all_windows)
    normal_cols = df.drop(normal_cols,axis=1).columns
    cols_to_return.append(list(normal_cols))
    
    # append the cols groups depending on the *cols_groups
    if col_groups != None:
        window_cols = np.array(df.columns)
        for group in col_groups:
            mask = np.where(np.char.find(window_cols.astype(str), group)==-1,False,True) # Mask filtering
            cols = window_cols[mask]
            cols_to_return.append(list(cols))
    
    return [k for i in cols_to_return for k in i]