import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from diagnostics import model_predictions



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config['output_model_path'])



##############Function for reporting
def score_model(input_data):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace

    x_data = input_data[['lastmonth_activity',
                        'lastyear_activity',
                         'number_of_employees']]
    y_data = input_data['exited']

    y_prediction = model_predictions(x_data)

    confusion_matrix_data = confusion_matrix(y_data, y_prediction)

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_data,
                                    display_labels = [0, 1])
    disp.plot()
    plt.savefig(model_path + '/confusionmatrix2.png')

if __name__ == '__main__':
    input_data = pd.read_csv(test_data_path + '/testdata.csv')
    score_model(input_data)
