from datetime import datetime
import os
from sklearn.linear_model import LinearRegression

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error

from .preprocessing import split_data, load_data_from_local_csv, preprocess
from google.cloud import storage

def build_model_using_data_from_cloud_storage():
    try:
        #Load data
        # Replace with your bucket name and model file name
        BUCKET_NAME = 'prebuilt-models'
        DATA_FILE = 'car-price-predictor/data/car_sales_data.csv'
        GCS_URI = f"gs://{BUCKET_NAME}/{DATA_FILE}"
        print(GCS_URI)

        df = load_data_from_local_csv(GCS_URI)
        
        print("Data frame loaded")
        print(df.shape)

        #Preprocess / Encode categorical variables Model and Fuel Type
        df = preprocess(df)
        print(df.shape)

        #Feature Engineering / training and test data splitting.
        X_train, X_test, y_train, y_test = split_data(df = df, target_column= "Price", test_size = 0.2, random_state = 7)
        print("Training and testing data split done.")

        model = build_model(X_train, y_train)
        print("model building done.")

        rmse = evaluate_model(model, X_test, y_test)
        print(f"Root mean square error for the model is: [{rmse}].")

        export_model_jobllib(model)
        print(f"Model successfully exported.")

        upload_model(model)
        print(f"Model successfully uploaded.")

        return "successfully completed."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def build_model(X_train, y_train):
    # Linear Regression
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    return rmse

def export_model_jobllib(model):
    try:
        file_path = "model.joblib"
        if os.path.exists(file_path):
            print("Existing file found. Deleting...")
            
            # Delete the file using os.remove()
            os.remove(file_path)
            print("Successfully deleted existing file.")
        else:
            print("No existing file found.")

        joblib.dump(model, file_path) # Write locally first

    except Exception as e:
        return f"An unexpected error occurred: {e}"

def upload_model(model):
    try:
        source_file_name = ".\model.joblib"
        MODEL_PATH = 'car-price-predictor/python-models'
        #GCS_ARTIFACT_PATH = f'gs://{BUCKET_NAME}/{MODEL_PATH}/model.joblib'
        GCS_ARTIFACT_PATH = "model.joblib"
        upload_blob("gs://prebuilt-models/car-price-predictor/python-models", source_file_name, GCS_ARTIFACT_PATH)    
    except Exception as e:
        return f"An unexpected error occurred: {e}"
    
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )
