import pandas as pd
import numpy as np
from os import listdir
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, recall_score, precision_score

from src import load_data as ld
from src import utils
from src import feature_exploring as fexp
from src import map_plot as mp
from src import ml_utils as mlu

import warnings
warnings.filterwarnings("ignore")

#### Load data

# load the train data
df_all_gears = pd.read_csv('data/train_by_mmsi.csv', index_col=[0])

# droping all the nul values (in the future, predict them)
df_all_gears_na = df_all_gears.dropna()

# converting the target into binary
df_all_gears_na['is_fishing'] = mlu.set_threshold(df_all_gears_na,0.5)

# selecting the cols needed for this model
    # Note: after several tests, I decided to use only the main features to train the model
    # Note 2: replace or drop window_1800? is maybe overfitting?
cols = fexp.column_select(df_all_gears_na,drop_always=True,drop_gear=True,col_groups=['3600','10800','21600','43200','86400'])
df_all_gears_na = df_all_gears_na[cols]

# taking a sample
# df_all_gears_s = df_all_gears_na.sample(400000,random_state=45) 

X = df_all_gears_na.drop('is_fishing',axis=1)
y = df_all_gears_na['is_fishing']

# train test split
X_train, X_test, y_train, y_test = mlu.train_test_mmsi_split(X,y,0.75) 

# saving mmsi numbers to plot later
# mmsi_train, mmsi_test = X_train['mmsi'], X_test['mmsi'] 
X_train.drop('mmsi', inplace=True, axis=1)
X_test.drop('mmsi', inplace=True, axis=1)

# scaling the train data and using the same object to scale the test data
X_train['distance_from_port'], X_test['distance_from_port'] = mlu.min_max_scaler(X_train[['distance_from_port']], 
                                                                             X_test[['distance_from_port']])

X_train['distance_from_shore'], X_test['distance_from_shore'] = mlu.min_max_scaler(X_train[['distance_from_shore']], 
                                                                             X_test[['distance_from_shore']])

#### GridSearchCV

'''
after several tests, I realized that random forest classifier gives the better results
'''

param_grid = {'n_estimators': [80,100], 
              'max_features': [7,10],
              'min_samples_split': [50,100],
              'n_jobs':[-1],
              'max_depth':[8,10]}
 
classifier = RandomForestClassifier()
grid_search = GridSearchCV(classifier, param_grid, cv=5,
                          scoring='f1')

#### Train the model

grid_search.fit(X_train, y_train)

#### Predict, show results and pickle the best model

best_classifier = grid_search.best_estimator_

# feature importances
feature_importances = {k:v*100 for k,v in zip(X_train.columns,best_classifier.feature_importances_)}
feature_importances_sorted = [(i,feature_importances[i]) for i in sorted(feature_importances,key=feature_importances.get, reverse=True)]

# predict
y_pred_train = best_classifier.predict(X_train)
y_pred_test = best_classifier.predict(X_test)

results = {'Accuracy': [accuracy_score(y_test, y_pred_test), accuracy_score(y_train, y_pred_train)],
           'Recall': [recall_score(y_test, y_pred_test), recall_score(y_train, y_pred_train)],
           'F1': [f1_score(y_test, y_pred_test), f1_score(y_train, y_pred_train)],
           'Parameters': best_classifier.get_params(),
           'set': ['test', 'train'],
          }

pickle.dump(best_classifier, open('results/grid_search_best_rf.pkl', 'wb'))       

#### Save results on a txt file

output_file = 'results/results.txt'

with open(output_file,'w') as f:

    f.write(f'Accuracy | test: {accuracy_score(y_test, y_pred_test)}, train: {accuracy_score(y_train, y_pred_train)}' + '\n')
    f.write(f'Recall | test: {recall_score(y_test, y_pred_test)}, train: {recall_score(y_train, y_pred_train)}' + '\n')
    f.write(f'F1 | test: {f1_score(y_test, y_pred_test)}, train: {f1_score(y_train, y_pred_train)}' + '\n')
    
    f.write(f'Best parameters: {best_classifier.get_params()}' + '\n')
    f.write(f'Feature importances: ' + '\n')
    
    for i in feature_importances_sorted:
        f.write(f'{i}' + '\n')


    




