# Purpose
This project is to create a end to end working demo for machine learning with google cloud.

This dataset contains detailed information on car sales, covering multiple manufacturers and models. It is suitable for data analysis. 

Here we use it for price prediction

# Regarding the App

1. Code builds different FLask APIs nammely for
2. Model Training:
   - To train model using local data (from github repo / docker container).
   - This will also export the model.
   - Model will be usable only after container is recreated.
3. Predictions:
   - Loads model from local data (from github repo / docker container)
   - Predict the price for raw user input data.
   - code will take care of preprocessing the raw data in order to predict the price.
4. Application is containerized. So Cloud Run is used to deploy it on GCP.
5. Application also has APIs to load pretrained model from GCP and then use it for prediction. 

## Generic Endpoints:

1. Healthcheck:
   - On GCP:
     ```code
      https://car-price-prediction-ml-model-89883393347.asia-south1.run.app
     ```
     
   - On local / debug mode:
     ```code
      http://127.0.0.1:8080
     ```
     
   - On local using container image:
     ```code
      Insider folder "car-price-prediction-ml-model"
      docker build -t car-price-predictor .
      docker run -p 8080:8080 car-price-predictor
      http://127.0.0.1:8080
     ```

2. Prediction Model Used:
   - This endpoint tells the number of models present for use in deployed app.
   - It also let us know the model loaded to make the prediction.
   - dir is a input query string parameter indicating the name os the models directory used to search for the models.
   - Pattern used for matching the model file is - "*_model_*.pkl" 
     ```code
     https://car-price-prediction-ml-model-89883393347.asia-south1.run.app/ModelFound?dir=models
     ``` 
    
## Endpint 1: Local Data + Local Model + Prediction APIs with Cloud Run

1. Train Model:
   - This uses training data from the local path (data/car_sales_data.csv)
   - Trained model will be preserved in models folder.
   - Since we do not attach volume, it will not be preserved / usable once container stops.
   - But this is fine, asmodel building is not expected to happpen on container. but it is expected to be a local activity (by dev on his/her machine)
   - To build model on local machine,
     ```code
      http://127.0.0.1:8080/build_model_locally
     ```
     
   - To build model on local machine with container (useless, as model will not be preserved)
     ```code
      Insider folder "car-price-prediction-ml-model"
      docker build -t car-price-predictor .
      docker run -p 8080:8080 car-price-predictor
      http://127.0.0.1:8080/build_model_locally
     ```
     
   - To build model on Cloud Run endpoint (uselss, as model will not be preserved),
     ```code
      https://car-price-prediction-ml-model-89883393347.asia-south1.run.app/build_model_locally
     ```
     
3. Predict:
   - Example 1 Actual Data point in training data: Manufacturer: BMW (unused feature), Model: M5, Engine size: 5, Fuel type: Petrol, Year of manufacture: 2010,
     Mileage: 58036	Price: 55476 (Target)
     ```code
     https://car-price-prediction-ml-model-89883393347.asia-south1.run.app/predict_local?
     Engine%20size=5&Year%20of%20manufacture=2010&Mileage=58000&Model=M5&Fuel%20type=Petrol
     ```
   - Example 2 Actual Data point in training data: Manufacturer: Toyota (unused feature), Model: RAV4, Engine size: 2.2, Fuel type: Hybrid, Year of manufacture:
     2008, Mileage: 68951	Price: 22064 (Target)
     ```code
     https://car-price-prediction-ml-model-89883393347.asia-south1.run.app/predict_local?
     Engine%20size=2.2&Year%20of%20manufacture=2008&Mileage=69000&Model=RAV4&Fuel%20type=Hybrid
     ```

## Endpint 2: Pretrained Model in GCS + Prediction APIs with Cloud Run

### Running App locally - 
1. Need to install Google Cloud SDK.
2. From Google Cloud SDK Shell run Cloud init and gcloud auth application-default login
3. Set default account to use and project.
4. In VSCode add plugin Cloud Code - restart - then VS Code terminal will allow running gcloud commands
5. Run gscloud init to set default account and project from VSCode terminal.
6. Then run the app.py python file as usual. 

### Running Container Locally
1. Create a Service Account.
   ```code
    gcloud iam service-accounts create download-gcs-file --display-name="Service Account to Download GCS Files"
   ```
3. Grand access to access cloud storgae.
   ```code
    gcloud projects add-iam-policy-binding nimble-octagon-253816 --member="serviceAccount:download-gcs-file@nimble-octagon-253816.iam.gserviceaccount.com" --role="roles/storage.objectViewer"
   ```
   -p => to map port, so all 8080 requests on machine will be trasfered to docker container.
   -e => to set environment variable
   -v => to map storage volume 
5. Create / download json key file. - Go to gcp >> IAM >> Service Account >> Keys >> Add New - AS JSON. It will download to machine.
6. Mount the key file (volume) whuile runnin docker run -v <Disk file path>:<docker file path>
7. set GOOGLE_APPLICATION_CREDENTIALS to file path.

### Preconditions

1. Model is built and ready to make predictions.
2. Exported model .pkl file is uploaded to Google Cloud Storage.
3. Service account with view access (storage-Object-Viewer) permissions is in place.
4. json key for authenticating this service account is created, downloaded, volume mapped and env. var is set before running the container.

### Predictions

1. Predict:
   - Example 1 Actual Data point in training data: Manufacturer: BMW (unused feature), Model: M5, Engine size: 5, Fuel type: Petrol, Year of manufacture: 2010,
     Mileage: 58036	Price: 55476 (Target)
     ```code
     https://car-price-prediction-ml-model-89883393347.asia-south1.run.app/predict_gcs?
     Engine%20size=5&Year%20of%20manufacture=2010&Mileage=58000&Model=M5&Fuel%20type=Petrol
     ```
   - Example 2 Actual Data point in training data: Manufacturer: Toyota (unused feature), Model: RAV4, Engine size: 2.2, Fuel type: Hybrid, Year of manufacture:
     2008, Mileage: 68951	Price: 22064 (Target)
     ```code
     https://car-price-prediction-ml-model-89883393347.asia-south1.run.app/predict_gcs?
     Engine%20size=2.2&Year%20of%20manufacture=2008&Mileage=69000&Model=RAV4&Fuel%20type=Hybrid
     ```


# References
1. Details regarding Exploratory Data Analysis can be found here: https://www.kaggle.com/code/shmagibokuchava7/pricepredict-pro
2. Setting up Application Default Credentials: https://cloud.google.com/docs/authentication/provide-credentials-adc#how-to
3. Connecting VS Code to GCP: https://www.youtube.com/watch?v=KeZ2ncbVPyE

# Dataset
https://www.kaggle.com/datasets/msnbehdani/mock-dataset-of-second-hand-car-sales











