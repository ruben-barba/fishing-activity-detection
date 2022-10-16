import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os import listdir
import folium
import pickle
import random

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score,recall_score, f1_score, confusion_matrix, precision_score

from src import load_data as ld
from src import utils
from src import feature_exploring as fexp
from src import map_plot as mp
from src import ml_utils as mlu

import warnings
warnings.filterwarnings("ignore")

#### loading and preparing the new data
'''
The script is using the validation data to predict and show plots. This data is unseen to the model.
'''

# loading the pickeled model
pickled_model = pickle.load(open('results/grid_search_best_rf.pkl', 'rb'))

# loading the new data
new_data = pd.read_csv('data/new_data/validation_by_mmsi.csv',index_col=[0])

# droping all the nul values (in the future, predict them)
new_data_na = new_data.dropna()

# setting a threshold to is_fishing
new_data_na['is_fishing'] = mlu.set_threshold(new_data_na,0.5)

# save mmsi col
mmsi_col = new_data_na['mmsi']

# columns to keep for the prediction
cols = fexp.column_select(new_data_na,drop_always=True,drop_mmsi=True,drop_y=True,drop_gear=True,col_groups=['3600','10800','21600','43200','86400'])
new_data_cols = new_data_na[cols]

# scaling the train data and using the same object to scale the test data
new_data_cols['distance_from_port'] = utils.min_max(new_data_cols,'distance_from_port')
new_data_cols['distance_from_shore'] = utils.min_max(new_data_cols,'distance_from_shore')

#### the prediction

prediction = pickled_model.predict(new_data_cols)

#### the results

result = pd.DataFrame({'mmsi':mmsi_col,
                       'is_fishing':new_data_na['is_fishing'],
                       'prediction':prediction,
                       'lat':new_data_na['lat'],
                       'lon':new_data_na['lon'],
                       'gear_type':new_data_na['gear_type']
                      })

y_test = list(result['is_fishing'])
y_pred_test = list(result['prediction'])

scores = {'Accuracy': accuracy_score(y_test, y_pred_test),
          'Precision': precision_score(y_test, y_pred_test),
          'Recall': recall_score(y_test, y_pred_test),
          'F1': f1_score(y_test, y_pred_test),
          'set': 'test'}

# assign green to 1 and red to 0
result['color_is_fishing'] = np.where(result['is_fishing']==1,'green','red')
result['color_prediction'] = np.where(result['prediction']==1,'green','red')

# save the predictions of every single vessel on a list
vessel_results = []
vessels = result['mmsi'].unique()
for i in vessels:
    vessel_results.append({
        'mmsi':i,
        'Accuracy': accuracy_score(result[result['mmsi']==i]['is_fishing'], result[result['mmsi']==i]['prediction']),
        'Recall': recall_score(result[result['mmsi']==i]['is_fishing'], result[result['mmsi']==i]['prediction']),
        'F1': f1_score(result[result['mmsi']==i]['is_fishing'], result[result['mmsi']==i]['prediction'],zero_division=0),
        'gear_type': result[result['mmsi']==i]['gear_type'].unique()[0]
                          })

# saving plots of all vessels 
for mmsi,gear in result.groupby('mmsi')['gear_type'].max().items():
    
    latitudes = np.array(result[result.mmsi==mmsi]['lat'])
    longitudes = np.array(result[result.mmsi==mmsi]['lon'])
    coords = [(lat,lon) for lat,lon in zip(latitudes,longitudes)]
    
    m_real = mp.folium_markers_real(result,mmsi)
    m_pred = mp.folium_markers_pred(result,mmsi)
    
    m_real.save(f'results/plots/{gear}_{mmsi}_real.html')
    m_pred.save(f'results/plots/{gear}_{mmsi}_pred.html')
    
# save the results in a txt
output_file = 'results/results_by_vessel.txt'

with open(output_file,'w') as f:
    for i in vessel_results:
        f.write(f'{i}'+'\n')
