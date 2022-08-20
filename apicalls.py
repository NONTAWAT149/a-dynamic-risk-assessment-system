#import requests
import subprocess
import os
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1/"

def call_api():

    #Call each API endpoint and store the responses
    response1 = subprocess.run(['curl', URL[:-1] +':8000/prediction?file_name=testdata/testdata.csv'], capture_output = True).stdout
    response2 = subprocess.run(['curl', URL[:-1] +':8000/scoring'], capture_output = True).stdout
    response3 = subprocess.run(['curl', URL[:-1] +':8000/summarystats'], capture_output = True).stdout
    response4 = subprocess.run(['curl', URL[:-1] +':8000/diagnostics'], capture_output = True).stdout

    #combine all API responses
    responses = {'prediction': response1,
                 'f1-score': response2,
                 'summarystats': response3,
                 'diagnostics': response4
                }

    #write the responses to your workspace
    # write text file to export f1-score

    with open('config.json','r') as f:
        config = json.load(f)

    export_path = os.path.join(config['output_model_path']) + '/apireturns2.txt'


    with open(export_path, 'w') as file:
        file.write(str(responses))
    print('results exported')


if __name__ == "__main__":
    call_api()