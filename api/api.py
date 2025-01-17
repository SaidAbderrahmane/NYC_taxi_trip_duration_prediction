import os
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.encoders import jsonable_encoder
import mlflow
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import common
from model_training import train_model

app = FastAPI()

class PredictionData(BaseModel):
    hour: int
    month: int
    weekday: int

@app.get("/")
async def root():
    return {"message": "Hello World"}

def inverse_transform_target(y_pred):
    return np.expm1(y_pred)

@app.post("/predict")
async def predict(data: PredictionData):
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    model_name = "NYC_Taxi_Trip_Prediction"

    # Get all registered versions of the model
    versions = client.search_model_versions(f"name='{model_name}'")

    # Find the latest version
    latest_version = max(int(v.version) for v in versions)

    # Load the latest version
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")

    input_df = pd.DataFrame([{
        "hour": data.hour,
        "weekday": data.weekday,
        "month": data.month
    }])
    input_df['hour'] = pd.to_numeric(input_df['hour'], downcast='integer')
    input_df['weekday'] = pd.to_numeric(input_df['weekday'], downcast='integer')
    input_df['month'] = pd.to_numeric(input_df['month'], downcast='integer')
    
    predictions = inverse_transform_target(model.predict(input_df))
    return JSONResponse(content=predictions.tolist(), media_type="application/json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)