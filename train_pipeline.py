import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# ---------------- CONFIG ----------------
DATA_PATH = "dataset_logistica_ml_10k.csv"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "ritardo"
NUMERIC = [
    "distanza_km", "valore_merce_eur", "peso_kg",
    "numero_transiti", "rischio_meteo", "rischio_doganale"
]
CATEGORICAL = ["modalità_trasporto", "fragile", "tracking_gps"]

# ---------------- LOAD DATA ----------------
df = pd.read_csv(DATA_PATH)
X = df[NUMERIC + CATEGORICAL]
y = df[TARGET]

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ---------------- PIPELINE ----------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
])

rf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

ensemble = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb)],
    voting="soft"
)

pipeline = Pipeline([
    ("prep", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("clf", ensemble)
])

# ---------------- TRAIN ----------------
pipeline.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
print(f"ROC-AUC: {auc:.4f}")

# ---------------- SAVE ----------------
joblib.dump(pipeline, MODEL_DIR / "logistic_guardian.pkl")
print("Model saved ✔")

#cd D:\Github_code_back\esercizi-python-rajaroybca6\logistic_project
#streamlit run train_pipeline.py