from training import train_model
import deployment
import diagnostics
from reporting import score_model
from ingestion import merge_multiple_dataframe
from apicalls import call_api

import json
import os
import pandas as pd

with open('config.json','r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
input_folder_path = config['input_folder_path']


def full_process():

    new_data_flag = 0
    retrain_flag = 0
    model_drift_flag = 0
    redeploy_flag = 0

    ##################Check and read new data
    #first, read ingestedfiles.txt
    file_path = os.getcwd() + '/' + prod_deployment_path + '/ingestedfiles.txt'
    with open(file_path, 'r') as file:
        txt_file = file.readlines()
    file.close()

    exist_file_name = txt_file[1]
    exist_file_name = exist_file_name.replace('\n', '')
    exist_file_name = exist_file_name.replace('[', '')
    exist_file_name = exist_file_name.replace(']', '')
    exist_file_name = exist_file_name.replace('"', '')
    exist_file_name = exist_file_name.replace("'", "")
    exist_file_name = exist_file_name.replace(' ', '')
    exist_file_name = exist_file_name.split(',')

    #second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    file_name_list = os.listdir(os.getcwd() + '/' + input_folder_path)
    new_file = []
    for file_name in file_name_list:
        if file_name[-3:] == 'csv' and file_name not in exist_file_name:
            new_file.append(file_name)

    #run ingestion function to merge new data
    if len(new_file) > 0:
        merge_multiple_dataframe()
        new_data_flag = 1


    ##################Deciding whether to proceed, part 1
    #if you found new data, you should proceed. otherwise, do end the process here

    if new_data_flag == 1:
        train_model()
        retrain_flag = 1


    ##################Checking for model drift
    #check whether the score from the deployed model is different
    # from the score from the model that uses the newest ingested data
    if retrain_flag == 1:

        def get_model_score():
            file_path = os.getcwd() + '/' + prod_deployment_path + '/latestscore.txt'
            with open(file_path, 'r') as file:
                latest_score = file.readlines()
            file.close()
            return latest_score[-1]

        #read the latest score
        latest_score = get_model_score()

        #new score model is going to be updated.
        test_df = pd.read_csv(dataset_csv_path + '/finaldata.csv')
        score_model(test_df)
        new_score = get_model_score()

        #check if the model is drift.
        if new_score < latest_score:
            model_drift_flag = 1

    ##################Deciding whether to proceed, part 2
    #if you found model drift, you should proceed. otherwise, do end the process here

    ##################Re-deployment
    #if you found evidence for model drift, re-run the deployment.py script
    if model_drift_flag == 1:
        if retrain_flag == 0:
            train_model()
        store_model_into_pickle()
        redeploy_flag = 1


    ##################Diagnostics and reporting
    #run diagnostics.py and reporting.py for the re-deployed model

    if redeploy_flag == 1:

        #run confusion matrix analysis
        score_model()

        #enable api
        os.system('python app.py')

        #call api
        call_api()


if __name__ == '__main__':
    full_process()





