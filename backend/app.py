"""
app.py
Flask REST API that serves the trained heart disease risk model
and returns personalized recommendations based on the input factors.

Run with:
    python app.py

Then it's available at http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load("heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    FEATURE_ORDER = joblib.load("feature_order.pkl")
    print("Model, scaler, and feature order loaded successfully.")
except FileNotFoundError:
    raise RuntimeError(
        "Model files not found. Run 'python train_model.py' first "
        "to generate heart_model.pkl, scaler.pkl, feature_order.pkl."
    )


def build_recommendations(age_years, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active):
    """
    Rule-based, clinically-informed suggestions driven by which specific
    inputs are pushing risk up. Not a diagnosis — general lifestyle guidance.
    """
    recs = []

    # Blood pressure
    if ap_hi >= 140 or ap_lo >= 90:
        recs.append({
            "title": "Blood pressure is in the high range",
            "detail": "Readings at or above 140/90 fall in hypertensive range. Consider a follow-up with a doctor, reducing sodium intake, and regular home monitoring."
        })
    elif ap_hi >= 130 or ap_lo >= 85:
        recs.append({
            "title": "Blood pressure is elevated",
            "detail": "Not yet hypertensive, but trending high. Cutting back on salt and caffeine, and rechecking in a few weeks, is a reasonable next step."
        })

    # BMI
    if bmi >= 30:
        recs.append({
            "title": "BMI is in the obese range",
            "detail": f"BMI of {bmi:.1f} is associated with higher cardiovascular risk. A gradual, sustainable weight loss plan with a doctor or dietitian is worth discussing."
        })
    elif bmi >= 25:
        recs.append({
            "title": "BMI is in the overweight range",
            "detail": f"BMI of {bmi:.1f} is above the healthy range (18.5-24.9). Small, consistent changes in diet and activity tend to help most."
        })
    elif bmi < 18.5:
        recs.append({
            "title": "BMI is below the healthy range",
            "detail": f"BMI of {bmi:.1f} is underweight. Worth discussing with a doctor to rule out underlying causes."
        })

    # Cholesterol
    if cholesterol == 3:
        recs.append({
            "title": "Cholesterol is well above normal",
            "detail": "Consider a lipid panel follow-up and a diet lower in saturated fat. This is one of the strongest modifiable risk factors."
        })
    elif cholesterol == 2:
        recs.append({
            "title": "Cholesterol is above normal",
            "detail": "Slightly elevated cholesterol can often be improved with diet, exercise, and periodic monitoring."
        })

    # Glucose
    if gluc == 3:
        recs.append({
            "title": "Glucose is well above normal",
            "detail": "This can indicate a higher diabetes risk, which independently raises cardiovascular risk. Worth discussing screening with a doctor."
        })
    elif gluc == 2:
        recs.append({
            "title": "Glucose is above normal",
            "detail": "Mildly elevated glucose is worth tracking over time, alongside diet and activity habits."
        })

    # Smoking
    if smoke == 1:
        recs.append({
            "title": "Smoking is a major risk factor",
            "detail": "Smoking is one of the single largest contributors to cardiovascular disease risk. Quitting has measurable benefits within months."
        })

    # Alcohol
    if alco == 1:
        recs.append({
            "title": "Alcohol intake noted",
            "detail": "Reducing regular alcohol intake can help lower blood pressure and overall cardiovascular risk over time."
        })

    # Activity
    if active == 0:
        recs.append({
            "title": "Low physical activity",
            "detail": "Even 150 minutes of moderate activity a week (brisk walking counts) meaningfully lowers cardiovascular risk."
        })

    # Age context (informational, not actionable)
    if age_years >= 55:
        recs.append({
            "title": "Age is a non-modifiable risk factor",
            "detail": "Risk naturally increases with age — this makes managing the modifiable factors above more impactful, not less."
        })

    if not recs:
        recs.append({
            "title": "No major risk factors flagged",
            "detail": "Vitals and lifestyle inputs look within normal ranges. Keep up regular checkups and healthy habits."
        })

    return recs


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "message": "Heart Disease Predictor API is running.",
        "endpoints": {"/predict": "POST - send patient data, get a prediction + recommendations"}
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        required_fields = [
            "age", "gender", "height", "weight",
            "ap_hi", "ap_lo", "cholesterol", "gluc",
            "smoke", "alco", "active"
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        age_years = float(data["age"])
        gender = int(data["gender"])
        height = float(data["height"])
        weight = float(data["weight"])
        ap_hi = float(data["ap_hi"])
        ap_lo = float(data["ap_lo"])
        cholesterol = int(data["cholesterol"])
        gluc = int(data["gluc"])
        smoke = int(data["smoke"])
        alco = int(data["alco"])
        active = int(data["active"])

        if not (80 <= ap_hi <= 250):
            return jsonify({"error": "Systolic BP (ap_hi) must be between 80 and 250"}), 400
        if not (40 <= ap_lo <= 200):
            return jsonify({"error": "Diastolic BP (ap_lo) must be between 40 and 200"}), 400
        if ap_hi < ap_lo:
            return jsonify({"error": "Systolic BP cannot be lower than diastolic BP"}), 400
        if not (120 <= height <= 220):
            return jsonify({"error": "Height must be between 120 and 220 cm"}), 400
        if not (30 <= weight <= 200):
            return jsonify({"error": "Weight must be between 30 and 200 kg"}), 400

        # ---------------------------------------------
        # Same engineered features used at training time
        # ---------------------------------------------
        bmi = weight / ((height / 100) ** 2)
        pulse_pressure = ap_hi - ap_lo
        map_val = ap_lo + (ap_hi - ap_lo) / 3
        age_bmi = age_years * bmi
        chol_gluc = cholesterol * gluc

        feature_map = {
            "age_years": age_years,
            "gender": gender,
            "height": height,
            "weight": weight,
            "bmi": bmi,
            "ap_hi": ap_hi,
            "ap_lo": ap_lo,
            "pulse_pressure": pulse_pressure,
            "map": map_val,
            "cholesterol": cholesterol,
            "gluc": gluc,
            "chol_gluc": chol_gluc,
            "smoke": smoke,
            "alco": alco,
            "active": active,
            "age_bmi": age_bmi,
        }
        features = np.array([[feature_map[f] for f in FEATURE_ORDER]])
        features_scaled = scaler.transform(features)

        prediction = int(model.predict(features_scaled)[0])
        probability = float(model.predict_proba(features_scaled)[0][1])

        risk_level = (
            "High" if probability >= 0.66 else
            "Moderate" if probability >= 0.33 else
            "Low"
        )

        recommendations = build_recommendations(
            age_years, bmi, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
        )

        return jsonify({
            "prediction": prediction,
            "probability": round(probability * 100, 2),
            "risk_level": risk_level,
            "bmi": round(bmi, 1),
            "recommendations": recommendations
        })

    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)