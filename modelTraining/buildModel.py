from datetime import datetime
import os
from sklearn.linear_model import LinearRegression

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error
from .preprocessing import split_data, load_data_from_local_csv, preprocess

def build_model_using_local_data():
    try:
        #Load data
        file_path = os.path.join("modelTraining", "data", "car_sales_data.csv")
        df = load_data_from_local_csv(file_path)
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

        export_model(model)
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

def export_model(model):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    filename = f".\models\liner_regression_model_{formatted_time}.pkl"
    joblib.dump(model, filename)
