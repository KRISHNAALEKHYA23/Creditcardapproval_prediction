"""
train_pipeline.py
------------------
End-to-end pipeline for the Credit Card Approval Prediction System.

Steps:
 1. Load application_record.csv + credit_record.csv
 2. Feature engineering: turn STATUS history into a binary TARGET
    (1 = high risk / reject, 0 = good applicant / approve)
 3. Clean & preprocess (missing values, duplicates, encoding)
 4. EDA plots saved to /static/plots
 5. Train Logistic Regression, Decision Tree, Random Forest, XGBoost
 6. Evaluate with accuracy / confusion matrix / classification report
 7. Save the best model + encoders + scaler to /model for the Flask app
"""

import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
PLOT_DIR = os.path.join(BASE_DIR, "static", "plots")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
app_df = pd.read_csv(os.path.join(DATA_DIR, "application_record.csv"))
credit_df = pd.read_csv(os.path.join(DATA_DIR, "credit_record.csv"))

print("Application record shape:", app_df.shape)
print("Credit record shape:", credit_df.shape)

# ---------------------------------------------------------------------
# 2. Feature engineering: build binary TARGET from payment STATUS
#    STATUS in {'2','3','4','5'} => overdue 60+ days => high risk (1)
#    everything else ('0','1','C','X') => acceptable (0)
# ---------------------------------------------------------------------
credit_df["IS_RISK"] = credit_df["STATUS"].isin(["2", "3", "4", "5"]).astype(int)
risk_by_id = credit_df.groupby("ID")["IS_RISK"].max().reset_index()
risk_by_id.rename(columns={"IS_RISK": "TARGET"}, inplace=True)
# TARGET = 1 -> applicant had a seriously delinquent month -> REJECT
# TARGET = 0 -> clean payment history -> APPROVE

df = app_df.merge(risk_by_id, on="ID", how="inner")
print("Merged dataset shape:", df.shape)
print("Target distribution:\n", df["TARGET"].value_counts(normalize=True))

# ---------------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------------
df.drop_duplicates(inplace=True)
df["OCCUPATION_TYPE"] = df["OCCUPATION_TYPE"].fillna("Unknown")

# Derived, human-friendly features
df["AGE_YEARS"] = (-df["DAYS_BIRTH"] // 365).astype(int)
df["YEARS_EMPLOYED"] = np.where(
    df["DAYS_EMPLOYED"] > 0, 0, (-df["DAYS_EMPLOYED"] // 365)
).astype(int)

feature_cols_numeric = [
    "CNT_CHILDREN", "AMT_INCOME_TOTAL", "AGE_YEARS", "YEARS_EMPLOYED",
    "CNT_FAM_MEMBERS", "FLAG_WORK_PHONE", "FLAG_PHONE", "FLAG_EMAIL",
]
feature_cols_categorical = [
    "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

model_df = df[feature_cols_numeric + feature_cols_categorical + ["TARGET"]].copy()

# ---------------------------------------------------------------------
# 4. EDA plots
# ---------------------------------------------------------------------
sns.set_style("darkgrid")

plt.figure(figsize=(5, 4))
sns.countplot(x="TARGET", data=model_df)
plt.title("Target distribution (0=Approve, 1=Reject)")
plt.savefig(os.path.join(PLOT_DIR, "target_distribution.png"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 4))
sns.histplot(model_df["AMT_INCOME_TOTAL"], bins=40, kde=True)
plt.title("Annual Income Distribution")
plt.savefig(os.path.join(PLOT_DIR, "income_distribution.png"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(y="NAME_INCOME_TYPE", hue="TARGET", data=model_df)
plt.title("Income Type vs Approval Outcome")
plt.savefig(os.path.join(PLOT_DIR, "income_type_vs_target.png"), bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(x="TARGET", y="AGE_YEARS", data=model_df)
plt.title("Age vs Outcome")
plt.savefig(os.path.join(PLOT_DIR, "age_vs_target.png"), bbox_inches="tight")
plt.close()

print("EDA plots saved to", PLOT_DIR)

# ---------------------------------------------------------------------
# 5. Encode categoricals
# ---------------------------------------------------------------------
encoders = {}
for col in feature_cols_categorical:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col].astype(str))
    encoders[col] = le

X = model_df.drop(columns=["TARGET"])
y = model_df["TARGET"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------------
# 6. Train & evaluate models
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        eval_metric="logloss", random_state=42,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1)
    ),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)
    results[name] = {"model": model, "accuracy": acc, "confusion_matrix": cm, "report": report}
    print(f"\n{name}: accuracy = {acc:.4f}")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))

# ---------------------------------------------------------------------
# 7. Pick best model & save everything the Flask app needs
# ---------------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_name]["model"]
print(f"\nBest model: {best_name} (accuracy={results[best_name]['accuracy']:.4f})")

with open(os.path.join(MODEL_DIR, "best_model.pkl"), "wb") as f:
    pickle.dump(best_model, f)
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, "encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)
with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
    json.dump(list(X.columns), f)

summary = {name: {"accuracy": r["accuracy"]} for name, r in results.items()}
summary["best_model"] = best_name
with open(os.path.join(MODEL_DIR, "model_comparison.json"), "w") as f:
    json.dump(summary, f, indent=2)

# model comparison bar chart
plt.figure(figsize=(6, 4))
names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
sns.barplot(x=accs, y=names)
plt.xlabel("Accuracy")
plt.title("Model Comparison")
plt.xlim(0, 1)
plt.savefig(os.path.join(PLOT_DIR, "model_comparison.png"), bbox_inches="tight")
plt.close()

print("\nSaved model artifacts to", MODEL_DIR)
print("Done.")
