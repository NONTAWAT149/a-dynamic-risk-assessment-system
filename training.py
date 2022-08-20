from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

###################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for training the model
def train_model():
    
    #use this logistic regression for training
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    
    #read dataset
    df = pd.read_csv(dataset_csv_path + '/finaldata.csv')
    x_data = df[['lastmonth_activity',
                'lastyear_activity',
                'number_of_employees']]
    y_data = df['exited']

    # fit the logistic regression to your data
    model.fit(x_data, y_data)
    
    #write the trained model to your workspace in a file called trainedmodel.pkl
    file_name = 'trainedmodel.pkl'

    #create model path if not exist
    if model_path not in os.listdir(os.getcwd()):
        os.makedirs(model_path)
        print('Model directory is created.')

    #export model
    pickle.dump(model, open(model_path + '/' + file_name, 'wb'))
    print('save model completed')


if __name__ == "__main__":
    train_model()