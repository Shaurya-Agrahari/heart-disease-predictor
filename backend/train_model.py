"""
train_model.py
Trains a heart disease (cardiovascular) risk classifier on cardio_train.csv
and saves the model + scaler for the Flask API to use.

Run this ONCE (or whenever you want to retrain):
    python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
print("Loading data...")
df = pd.read_csv("cardio_train.csv", sep=";")

# ---------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------
df["age_years"] = (df["age"] / 365).astype(int)
df.drop(["id", "age"], axis=1, inplace=True)

# ---------------------------------------------------------
# 3. Clean obviously invalid / outlier medical readings
# ---------------------------------------------------------
before = len(df)
df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 250)]
df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 200)]
df = df[df["ap_hi"] >= df["ap_lo"]]
df = df[(df["height"] >= 120) & (df["height"] <= 220)]
df = df[(df["weight"] >= 30) & (df["weight"] <= 200)]
print(f"Removed {before - len(df)} outlier/invalid rows ({len(df)} rows remain)")

# ---------------------------------------------------------
# 4. Extra engineered features (small but real accuracy gain)
# ---------------------------------------------------------
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]                       # gap between systolic/diastolic
df["map"] = df["ap_lo"] + (df["ap_hi"] - df["ap_lo"]) / 3              # mean arterial pressure
df["age_bmi"] = df["age_years"] * df["bmi"]                            # age/weight interaction
df["chol_gluc"] = df["cholesterol"] * df["gluc"]                       # combined metabolic risk

# ---------------------------------------------------------
# 5. Feature order MUST match what app.py sends at inference time
# ---------------------------------------------------------
FEATURE_ORDER = [
    "age_years", "gender", "height", "weight", "bmi",
    "ap_hi", "ap_lo", "pulse_pressure", "map",
    "cholesterol", "gluc", "chol_gluc",
    "smoke", "alco", "active", "age_bmi"
]

X = df[FEATURE_ORDER]
y = df["cardio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 6. Scale features
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 7. Train model (LightGBM, tuned via RandomizedSearchCV separately —
#    best found params baked in here)
# ---------------------------------------------------------
print("Training model...")
model = LGBMClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.6,
    colsample_bytree=0.9,
    num_leaves=15,
    min_child_samples=40,
    random_state=42,
    verbosity=-1
)
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------
# 8. Evaluate honestly
# ---------------------------------------------------------
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print(f"TEST ACCURACY: {acc * 100:.2f}%")
print("=" * 50)
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print(
    "\nNote: ~73-74% is close to the practical ceiling for this dataset with "
    "these features (physical exam + lifestyle data only, no bloodwork/ECG). "
    "This matches published benchmarks on this dataset."
)

# ---------------------------------------------------------
# 9. Save model + scaler + feature order for the API to use
# ---------------------------------------------------------
joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURE_ORDER, "feature_order.pkl")

print("\nSaved: heart_model.pkl, scaler.pkl, feature_order.pkl")
print("You can now run: python app.py")