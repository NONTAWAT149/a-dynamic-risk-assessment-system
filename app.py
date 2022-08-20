from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
#import create_prediction_model
#import diagnosis
#import predict_exited_from_saved_model
import json
import os

from diagnostics import model_predictions, dataframe_summary, execution_time, missing_data_check, outdated_packages_list
from scoring import score_model

######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def predict():        
    #call the prediction function you created in Step 3
    input_data = request.args.get('file_name')
    input_data = pd.read_csv(input_data)

    x_data = input_data[['lastmonth_activity',
                        'lastyear_activity',
                         'number_of_employees']]

    prediction = model_predictions(x_data)
    return str(prediction) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():
    #check the score of the deployed model

    f1_score = score_model()
    return str(f1_score) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    #check means, medians, and modes for each column

    statistic_data = dataframe_summary()
    return str(statistic_data) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    #check timing and percent NA values
    time_run_list = execution_time()
    missing_data = missing_data_check()
    package = outdated_packages_list()

    # add return value for all diagnostics
    return {'time_execution': time_run_list,
            'missing_data': missing_data,
            'package': package
            }

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
