from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

model = joblib.load("models/logistic_guardian.pkl")

app = FastAPI(title="Logistic Guardian API")

class Shipment(BaseModel):
    distanza_km: float
    valore_merce_eur: float
    peso_kg: float
    numero_transiti: int
    rischio_meteo: float
    rischio_doganale: float
    modalità_trasporto: str
    fragile: str
    tracking_gps: str

@app.post("/predict")
def predict(data: Shipment):
    df = pd.DataFrame([data.dict()])
    proba = model.predict_proba(df)[0][1]
    return {
        "delay_probability": round(float(proba), 4),
        "delay": int(proba > 0.5)
    }
