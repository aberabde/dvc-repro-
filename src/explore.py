import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn import metrics
import os
 

df = pd.read_csv('data/Ad.csv', index_col=0)

# create folder to save file
data_path = os.path.join('data', 'Exploration')
os.makedirs(data_path, exist_ok=True)
out = df.describe()
out.to_csv(os.path.join(data_path, "nblignes.csv"))
