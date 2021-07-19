
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn import metrics
import json
import os
import pickle
import yaml
import sys


model_filename = sys.argv[1]
Fitest_path = sys.argv[2]
scores_file = sys.argv[3]

#Scores_path = os.path.join('data', 'R2')
#os.makedirs(Scores_path, exist_ok=True)

features_test_pkl = os.path.join(Fitest_path, 'X_test.pkl')
with open(features_test_pkl, 'rb') as x:
    X_test = pickle.load(x)

target_test_pkl = os.path.join(Fitest_path, 'y_test.pkl')
with open(target_test_pkl, 'rb') as y:
    y_test = pickle.load(y)


# load model
with open(model_filename, 'rb') as f:
    model = pickle.load(f)


y_pred = model.predict(X_test)

import numpy as np
# RMSE
mse = np.sqrt(metrics.mean_squared_error(y_test,
                                        y_pred))

#Calcul du R-squared
r2 = metrics.r2_score(y_test, y_pred)    

with open(scores_file, 'w') as outfile:
    json.dump({ "MSE": mse, "R2":r2}, outfile)                                  
    
