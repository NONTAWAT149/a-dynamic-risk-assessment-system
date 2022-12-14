from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import subprocess


##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 


####################function for deployment
def store_model_into_pickle():
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory

    #create production directory if not exist
    if prod_deployment_path not in os.listdir(os.getcwd()):
        os.makedirs(prod_deployment_path)
        print('Model directory is created.')

    #copy model file
    model_path = os.path.join(config['output_model_path'])
    subprocess.run(['cp',
                    model_path + '/trainedmodel.pkl',
                    prod_deployment_path + '/trainedmodel.pkl'],
                   capture_output = True).stdout

    #copy latestscore.txt
    model_path = os.path.join(config['output_model_path'])
    subprocess.run(['cp',
                    model_path + '/latestscore.txt',
                    prod_deployment_path + '/latestscore.txt'],
                   capture_output = True).stdout

    #copy ingestfiles.txt
    subprocess.run(['cp',
                    dataset_csv_path + '/ingestedfiles.txt',
                    prod_deployment_path + '/ingestedfiles.txt'],
                   capture_output = True).stdout

    print('copy files for production is completed')


if __name__ == "__main__":
    store_model_into_pickle()
        
        

