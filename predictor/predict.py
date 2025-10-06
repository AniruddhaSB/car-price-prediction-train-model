import glob
import io
import os
import pickle
import joblib

from modelTraining.preprocessing import preprocess
from google.cloud import storage

def predict_car_price_using_pretrained_model(model, userInput_df):
    try:
        if model is not None:

            #Added dummy output column before preprocessing.
            userInput_df['Price'] = 0
            print(f"User input dataframe: {userInput_df}")

            preprocessed_df = preprocess(userInput_df)

            #Removed dummy output column beforeprediction.
            preprocessed_df.drop('Price', axis=1, inplace=True)

            print(preprocessed_df)
            print(f"Preprocessed dataframe: {userInput_df}")

            prediction = model.predict(preprocessed_df)
            print(f"Prediction: {prediction}")
            return f"Predicted price of a car: {prediction[0]:.2f}"
        else:
            return "Model not loaded, returning default - predicted car price as $0"
    except Exception as e:
        return f"An unexpected error occurred: {e}"



def load_model_from_local():
    # Use forward slashes for cross-platform compatibility
    search_path = os.path.join("models", "*_model_*.pkl")
    print(f"Searching for models at: {search_path}")

    # Use a sorted list to get the first one alphabetically.
    list_of_files = sorted(glob.glob(search_path))

    # Add a check to see if any files were found.
    if not list_of_files:
        raise FileNotFoundError(f"No model files found matching the pattern '{search_path}'")

    lastest_file = list_of_files[-1]
    print(f"Found and loading the following model file: {lastest_file}")

    try:
        model = joblib.load(lastest_file)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load the model file '{lastest_file}': {e}")
    
    
# Replace with your bucket name and model file name
BUCKET_NAME = 'prebuilt-models'
MODEL_FILE = 'car-price-predictor/python-models/liner_regression_model_20250916160618.pkl'
#MODEL_FILE = 'liner_regression_model_20250916160618.pkl'

# Initialize GCS client
storage_client = storage.Client()

# Function to download and load the model
def load_model_from_cloud_storage():
    try:
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_FILE)
        model_bytes = blob.download_as_bytes()
        model_file_like_object = io.BytesIO(model_bytes)
        model = joblib.load(model_file_like_object)
        print(f"Loaded model from GCS: {MODEL_FILE}")
        return model
    except Exception as e:
        print(f"Failed to load the model file: {e}")
