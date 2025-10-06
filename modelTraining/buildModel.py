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
        BUCKET_NAME = 'prebuilt-models'
        MODEL_PATH = 'car-price-predictor/python-models'
        GCS_ARTIFACT_PATH = f'gs://{BUCKET_NAME}/{MODEL_PATH}/model.joblib'
        #joblib.dump(model, GCS_ARTIFACT_PATH)
        joblib.dump(model, f'gs://{BUCKET_NAME}/model.joblib')

        print("Running IAM write permission check...")
        has_access, message = check_bucket_write_access(BUCKET_NAME)

        print(has_access, message)

    except Exception as e:
        return f"An unexpected error occurred: {e}"

def check_bucket_write_access(bucket_name):
    """
    Checks if the authenticated user has permission to create objects 
    (i.e., write/dump the model file) in the GCS bucket.
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Test for the required permission to create a new object
        required_permission = 'storage.objects.create'
        permissions = bucket.test_iam_permissions([required_permission])
        
        if required_permission in permissions:
            return True, f"Success: User has the required '{required_permission}' permission."
        else:
            return False, (f"Failure: User LACKS the required '{required_permission}' permission on bucket '{bucket_name}'. "
                           "Ensure your IAM role includes 'Storage Object Creator' or 'Storage Admin'.")
            
    except gcp_exceptions.NotFound:
        return False, f"Failure: The bucket '{bucket_name}' does not exist or is inaccessible."
    except Exception as e:
        return False, f"Failure: Cannot verify permissions due to client error: {e}"
