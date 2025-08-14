from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd, joblib

app = FastAPI(title="California Housing — Regressor API")
model = joblib.load("artifacts/model.joblib")

class HouseData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(item: HouseData):
    df = pd.DataFrame([item.dict()])
    yhat = float(model.predict(df)[0])
    return {"predicted_price": yhat}
