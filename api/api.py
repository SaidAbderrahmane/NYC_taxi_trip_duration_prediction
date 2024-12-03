import os
import sys
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.encoders import jsonable_encoder
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

@app.post("/predict")
async def predict(data: PredictionData):
    model = common.load_model(common.MODEL_PATH)
    input_df = pd.DataFrame([{
        "hour": data.hour,
        "weekday": data.weekday,
        "month": data.month
    }])
    predictions = model.predict(input_df)
    return JSONResponse(content=predictions.tolist(), media_type="application/json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)