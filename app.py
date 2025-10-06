import glob
import os
from flask import Flask, request
from datetime import datetime

import pandas as pd

from modelTraining.buildModel import build_model_using_data_from_cloud_storage

app = Flask(__name__)

@app.route('/')
def hello_world():
    # Get the current date and time
    current_time = datetime.now()

    # Format the date and time as a string
    # %Y = Year, %m = Month, %d = Day, %H = Hour, %M = Minute, %S = Second
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Create the final response string
    response = f'Namaste. Current date and time is: {formatted_time}'

    return response

@app.route('/build_model_gcs')
def build_model_gcs():
    msg = build_model_using_data_from_cloud_storage()
    return msg

@app.route('/DataFound')
def check_if_model_exists():

    dir = request.args.get('dir')
    if not os.path.isdir(dir):
        msg = f"Error: Directory does not exist: {dir}"
    else:
        msg = f"Error: Directory exist: {dir}"

        search_path = os.path.join(dir, "*.csv")
        list_of_files = glob.glob(search_path)
        msg = f"Directory Found. Files matching pattern: {len(list_of_files)}"
        sorted_files = sorted(list_of_files)
        first_file = sorted_files[-1]
        msg = f"Directory Found. Files matching pattern: {len(list_of_files)}. File selected to load {first_file}"

    return msg

# This block must be at the same level of indentation as the import statement and app = Flask(__name__)
if __name__ == '__main__':
    # Get the port from the environment variable, defaulting to 5000 if not found
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)