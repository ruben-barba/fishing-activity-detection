import pandas as pd
import numpy as np

from os import listdir
from tqdm import tqdm

from src import feature_exploring as fexp
from src import ml_utils as mlu

def load_concat_by_gear(gear_type):
    '''
    function that takes a gear type as argument, load the corresponding npz files in data folder and concatenate 
    them on a unique dataframe
    
    :param gear_type: the gear type name to load and concatenate
    :type gear_type: string
    
    :return: 'dataframe' as result of concatenating all the files containing gear_type word
    '''
    print(f'\nLoading {gear_type}...\n')
    df = pd.DataFrame()   
    path = 'data/training_data/'
    for file in tqdm(listdir(path)):
        if gear_type in file:
            npz = np.load(path+file)
            dataframe_npz = pd.DataFrame(npz.get("x"))
            df = df.append(dataframe_npz)
    return df

def load_multiple(gear_type_list):
    '''
    function that allows to multiple load the npz files using using 'load_concat_by_gear' function
    
    :param gear_type_list: a list of gear_types
    :type gear_type_list: list
    
    :return: 'dataframe' as result of concatenating all the files containing the word in the list
    '''
    
    print('\n')
    print('\nConcatenating data...\n')
    return [load_concat_by_gear(i) for i in tqdm(gear_type_list)]

def load_data_model(train_size=0.8):
    '''
    this function load all the npz files into a one csv file and split the resulting file on a training dataset and a validation dataset.
    
    :param train_size: size of the train split
    :type train_size: float [0,1]
    
    :return: nothing
    '''
    # all the data is here
    path = 'data/training_data/' 

    # all the gear types listed here
    #gear_type_list = ['Drifting_longlines', 'Fixed_gear','Purse_seines','Trawlers','Trollers','Pole_and_line','Unknown']
    gear_type_list = ['Drifting_longlines','Purse_seines','Trawlers']
    # loading all dataframes and concatenating into one
    #drifting_df, fixed_gear_df, purse_df, trawlers_df, trollers_df, pole_df, unknown_df = load_multiple(gear_type_list)
    drifting_df, purse_df, trawlers_df = load_multiple(gear_type_list)
    #dfs = [drifting_df, fixed_gear_df, purse_df, trawlers_df, trollers_df, pole_df, unknown_df]
    dfs = [drifting_df, purse_df, trawlers_df]
    
    # lets label the data creating a new column called 'gear_type'
    for df,name in zip(dfs,gear_type_list):
        df['gear_type'] = name
    
    #df_all_gears = pd.concat([drifting_df, fixed_gear_df, purse_df, trawlers_df, trollers_df, pole_df, unknown_df], ignore_index=True)
    df_all_gears = pd.concat(dfs, ignore_index=True)
    
    # splitting into train and production validation data in order to keep a split of the data as unseen
    train, val = mlu.mmsi_split(df_all_gears,train_size)
    
    # saving both as csv
    train_percentage = int(round(train.shape[0]/df_all_gears.shape[0]*100,0))
    val_percentage = int(round(val.shape[0]/df_all_gears.shape[0]*100,0))

    train.to_csv('data/train_by_mmsi.csv')
    val.to_csv('data/new_data/validation_by_mmsi.csv') # the new data is stored at new_data folder

    print('\n'+f'Train split: {train_percentage}% | Validation split: {val_percentage}%'+'\n')
    
    print([(k,len(v)) for k,v in val.groupby(['gear_type'])['mmsi'].unique().items()])
    
    return