import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file

    # Initailise data frame
    input_df = pd.DataFrame(columns = ['corporation',
                                       'lastmonth_activity',
                                       'lastyear_activity',
                                       'number_of_employees',
                                       'exited'])
    # Check file in input directory
    file_name_list = os.listdir(os.getcwd() + '/' + input_folder_path)

    # Create output directory if it does not exist.
    if output_folder_path not in os.listdir(os.getcwd()):
        os.makedirs(output_folder_path)
        print('Output directory is created.')

    # Collect input data
    file_record = []
    for file_name in file_name_list:
        if file_name[-3:] == 'csv':
            new_df = pd.read_csv(input_folder_path + '/' + file_name)
            input_df = input_df.append(new_df)
            file_record.append(file_name)

    # Remove duplication
    input_df.drop_duplicates(inplace = True)

    # Produce output file
    input_df.to_csv(output_folder_path + '/finaldata.csv', index=False)

    # Create Log data
    source_location = input_folder_path
    output_file_name = file_record
    data_size = len(pd.read_csv(output_folder_path + '/finaldata.csv'))
    timestamp = datetime.now()
    all_record = [source_location, '\n',
                  output_file_name, '\n',
                  data_size, '\n',
                  timestamp, '\n']

    file = open(output_folder_path + '/ingestedfiles.txt', 'w')
    for element in all_record:
        file.write(str(element))

if __name__ == '__main__':
    merge_multiple_dataframe()
