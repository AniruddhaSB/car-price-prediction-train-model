import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_data_from_local_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess(df:pd.DataFrame):
    le_model = LabelEncoder()
    le_fuel = LabelEncoder()

    df["Model_encoded"] = le_model.fit_transform(df["Model"])
    df["Fuel_type_encoded"] = le_fuel.fit_transform(df["Fuel type"])
    
    features = ['Model_encoded', 'Engine size', 'Fuel_type_encoded', 'Year of manufacture', 'Mileage', 'Price']
    new_df = df[features]

    return new_df

def split_data(df, target_column, test_size, random_state):
    
    # Split features (X) and target (y)
    y = df.pop(target_column)
    X = df
    
    # Perform the train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test