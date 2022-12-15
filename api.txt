from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime as dt
import ipdb
import pandas as pd
from taxifare_model.interface.main import pred
website = FastAPI()

website.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)



@website.get("/")
def root():
    return {'greeting': 'Hello'}

@website.get("/predict")
def predict(pickup_datetime: str,  
            pickup_longitude: float,    
            pickup_latitude: float,    
            dropoff_longitude: float,   
            dropoff_latitude: float,  
            passenger_count: int):

    pickup_datetime = dt.strptime(pickup_datetime, '%Y-%m-%d %H:%M:%S')

    X_pred = pd.DataFrame(dict(
            key=["2013-07-06 17:18:00"],  # useless but the pipeline requires it
            pickup_datetime=[pickup_datetime.strftime('%Y-%m-%d %H:%M:%S UTC')],
            pickup_longitude=[float(pickup_longitude)],
            pickup_latitude=[float(pickup_latitude)],
            dropoff_longitude=[float(dropoff_longitude)],
            dropoff_latitude=[float(dropoff_latitude)],
            passenger_count=[int(passenger_count)]))


    fare_predict = pred(X_pred,stage='Production')

    return {'fare': float(fare_predict[0,0])}
