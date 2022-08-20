from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file

    #load model
    model_file = os.path.join(config['output_model_path']) + '/trainedmodel.pkl'
    loaded_model = pickle.load(open(model_file, 'rb'))

    #call test data
    df = pd.read_csv(test_data_path + '/testdata.csv')
    x_data = df[['lastmonth_activity',
                'lastyear_activity',
                'number_of_employees']]
    y_data = df['exited']

    #model evaluation
    y_prediction = loaded_model.predict(x_data)

    f1_score = metrics.f1_score(y_prediction, y_data)
    print(f1_score)

    #write text file to export f1-score
    export_path = os.path.join(config['output_model_path']) + '/latestscore.txt'
    with open(export_path, 'w') as file:
        file.write(str(f1_score))
    print('f1 score exported')

    return f1_score


if __name__ == "__main__":
    score_model()


