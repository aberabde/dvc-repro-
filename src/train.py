
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn import metrics
import json
import os
import pickle
import yaml
import sys

df = pd.read_csv('data/Ad.csv', index_col=0)

Fitest_path = os.path.join('data', 'Fitest')
os.makedirs(Fitest_path, exist_ok=True)





model_filename = sys.argv[1]


cols_predicteurs = ['TV','radio','newspaper']
#predicteurs
X = df[cols_predicteurs]
y = df.sales

params = yaml.safe_load(open('params.yaml'))['train']

test_size =params['test_size']
random_state = params['random_state']

 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, 
                                        y , test_size = test_size, random_state = random_state)



lm = LinearRegression()
lm.fit(X_train,y_train)


  

def save_pkl(test,filename):
    output_file = os.path.join(Fitest_path, filename)
    output = test
    with open(output_file, 'wb') as t:
        pickle.dump(output, t)
        
save_pkl(X_test,'X_test.pkl')
save_pkl(y_test,'y_test.pkl')


with open(model_filename, 'wb') as f:
    pickle.dump(lm, f)
    
    
