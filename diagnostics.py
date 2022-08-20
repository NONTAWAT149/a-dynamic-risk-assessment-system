
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess
import pickle

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['prod_deployment_path'])

##################Function to get model predictions
def model_predictions(input_data):
    #read the deployed model and a test dataset, calculate predictions

    #get test data
    #input_data = pd.read_csv(test_data_path + '/testdata.csv')

    #import model
    model_file = os.getcwd() + '/' + model_path + '/trainedmodel.pkl'
    with open(model_file, 'rb') as file:
        model = pickle.load(file)

    # return value should be a list containing all predictions
    return model.predict(input_data) #run prediction

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here

    #read dataset
    df = pd.read_csv(dataset_csv_path + '/finaldata.csv')

    #calculate statistic data
    statistic_data = {}
    for column_name in df.columns.to_list():
        if df[column_name].dtypes == 'int64':
            statistic_data[column_name] = {'mean': df[column_name].mean(),
                                           'median': df[column_name].median(),
                                           'std': df[column_name].std()}

    #export statistic data
    with open(dataset_csv_path + '/statistic.txt', 'w') as file:
        file.write(json.dumps(statistic_data))

    # return value should be a list containing all summary statistics
    return statistic_data

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py

    #check time of training process
    start_time = timeit.default_timer()
    os.system('python training.py')
    training_time = timeit.default_timer() - start_time

    #check time of ingestion process
    start_time = timeit.default_timer()
    os.system('python ingestion.py')
    ingestion_time = timeit.default_timer() - start_time

    # return a list of 2 timing values in seconds
    return [training_time, ingestion_time]

##################Function to check dependencies
def outdated_packages_list():
    #get a list of

    #to get current and latest version of python package
    #this assumes that the code is running in the production environment
    #no need to call requirements.txt
    outdate_list = subprocess.check_output(['pip', 'list', '--outdated'])

    #export information
    with open('outdate.txt', 'wb') as file:
        file.write(outdate_list)

    #print('outdate_list type', type(outdate_list))

    return outdate_list.decode("utf-8")

def missing_data_check():

    #import dataset
    df = pd.read_csv(dataset_csv_path + '/finaldata.csv')

    #find and calculate percentage of NA data
    missing_data = {}
    for column_name in df.columns.to_list():
        missing_data[column_name] = (df[column_name].isna().sum())*100/len(df)

    #export information
    with open(dataset_csv_path + '/missing_data.txt', 'w') as file:
        file.write(json.dumps(missing_data))
    return missing_data


if __name__ == '__main__':
    input_data = pd.read_csv(test_data_path + '/testdata.csv')
    input_data = input_data[['lastmonth_activity',
                        'lastyear_activity',
                         'number_of_employees']]
    model_predictions(input_data)
    dataframe_summary()
    execution_time()
    outdated_packages_list()
    missing_data_check()






    
