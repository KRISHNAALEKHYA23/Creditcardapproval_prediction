"""
app.py
------
Flask web application for the Credit Card Approval Prediction System.

Routes:
  /            Home page introduction
  /predict     User input form (GET) + prediction handling (POST)
  /result      Prediction result display

Run locally:
    pip install -r requirements.txt
    python train_pipeline.py      # generates model/ artifacts (once)
    python app.py
Then open http://127.0.0.1:5000
"""

import json
import os
import pickle

import numpy as np
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)

# ---------------------------------------------------------------------
# Load trained model + preprocessing artifacts once at startup
# ---------------------------------------------------------------------
with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
    model = pickle.load(f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(MODEL_DIR, "encoders.pkl"), "rb") as f:
    encoders = pickle.load(f)
with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
    FEATURE_COLUMNS = json.load(f)
try:
    with open(os.path.join(MODEL_DIR, "model_comparison.json")) as f:
        MODEL_INFO = json.load(f)
except FileNotFoundError:
    MODEL_INFO = {}

CATEGORICAL_OPTIONS = {
    col: list(le.classes_) for col, le in encoders.items()
}


def build_feature_vector(form):
    """Turn raw form input into a properly ordered/encoded feature vector."""
    row = {
        "CNT_CHILDREN": int(form["children"]),
        "AMT_INCOME_TOTAL": float(form["income"]),
        "AGE_YEARS": int(form["age"]),
        "YEARS_EMPLOYED": int(form["years_employed"]),
        "CNT_FAM_MEMBERS": int(form["family_members"]),
        "FLAG_WORK_PHONE": 1 if form.get("work_phone") == "yes" else 0,
        "FLAG_PHONE": 1 if form.get("phone") == "yes" else 0,
        "FLAG_EMAIL": 1 if form.get("email") == "yes" else 0,
        "CODE_GENDER": form["gender"],
        "FLAG_OWN_CAR": form["own_car"],
        "FLAG_OWN_REALTY": form["own_realty"],
        "NAME_INCOME_TYPE": form["income_type"],
        "NAME_EDUCATION_TYPE": form["education"],
        "NAME_FAMILY_STATUS": form["family_status"],
        "NAME_HOUSING_TYPE": form["housing_type"],
        "OCCUPATION_TYPE": form["occupation"],
    }

    encoded_row = []
    for col in FEATURE_COLUMNS:
        val = row[col]
        if col in encoders:
            le = encoders[col]
            # guard against unseen categories
            val = val if val in le.classes_ else le.classes_[0]
            val = le.transform([val])[0]
        encoded_row.append(val)

    arr = np.array(encoded_row, dtype=float).reshape(1, -1)
    return scaler.transform(arr)


@app.route("/")
def home():
    return render_template("index.html", model_info=MODEL_INFO)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", options=CATEGORICAL_OPTIONS)

    try:
        X = build_feature_vector(request.form)
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        approved = pred == 0  # TARGET=0 -> good applicant -> approve
        confidence = round(float(proba[0] if approved else proba[1]) * 100, 2)

        return render_template(
            "result.html",
            approved=approved,
            confidence=confidence,
            model_name=MODEL_INFO.get("best_model", "ML Model"),
        )
    except Exception as e:
        return render_template("result.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
