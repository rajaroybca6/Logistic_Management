import pandas as pd
import joblib
from scipy.stats import ks_2samp


baseline = pd.read_csv("dataset_logistica_ml_10k.csv")
model = joblib.load("models/logistic_guardian.pkl")


NUMERIC = ["distanza_km","valore_merce_eur","peso_kg","numero_transiti","rischio_meteo","rischio_doganale"]




def detect_drift(new_data: pd.DataFrame):
drift = {}
for col in NUMERIC:
stat, p = ks_2samp(baseline[col], new_data[col])
drift[col] = p < 0.05
return drift


# Example
# new_batch = pd.read_csv("new_shipments.csv")
# print(detect_drift(new_batch))