"""
generate_data.py
-----------------
Generates a synthetic but realistic Credit Card Approval dataset in the same
shape/schema as the well-known "Credit Card Approval Prediction" dataset
(application_record.csv + credit_record.csv), so the rest of the pipeline
(EDA -> preprocessing -> modeling -> Flask app) works exactly the same way
it would on the real dataset. If you have the real Kaggle dataset, just drop
application_record.csv and credit_record.csv into this /data folder and skip
this script.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 8000  # number of applicants

# ---------------------------------------------------------------------
# application_record.csv  -> one row per applicant, demographic/financial
# ---------------------------------------------------------------------
ids = np.arange(5000000, 5000000 + N)

gender = np.random.choice(["M", "F"], size=N, p=[0.45, 0.55])
own_car = np.random.choice(["Y", "N"], size=N, p=[0.4, 0.6])
own_realty = np.random.choice(["Y", "N"], size=N, p=[0.6, 0.4])
children = np.random.poisson(0.5, size=N).clip(0, 5)

income_type = np.random.choice(
    ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
    size=N, p=[0.55, 0.20, 0.13, 0.10, 0.02]
)

education = np.random.choice(
    ["Secondary / secondary special", "Higher education", "Incomplete higher",
     "Lower secondary", "Academic degree"],
    size=N, p=[0.65, 0.22, 0.07, 0.04, 0.02]
)

family_status = np.random.choice(
    ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
    size=N, p=[0.6, 0.18, 0.1, 0.08, 0.04]
)

housing_type = np.random.choice(
    ["House / apartment", "With parents", "Municipal apartment",
     "Rented apartment", "Office apartment", "Co-op apartment"],
    size=N, p=[0.7, 0.12, 0.08, 0.06, 0.03, 0.01]
)

# annual income correlated with income_type/education
base_income = np.random.lognormal(mean=10.8, sigma=0.45, size=N)
income_bump = pd.Series(income_type).map({
    "Working": 1.0, "Commercial associate": 1.25, "Pensioner": 0.65,
    "State servant": 1.05, "Student": 0.4
}).values
edu_bump = pd.Series(education).map({
    "Secondary / secondary special": 1.0, "Higher education": 1.3,
    "Incomplete higher": 1.1, "Lower secondary": 0.85, "Academic degree": 1.5
}).values
annual_income = (base_income * income_bump * edu_bump).round(2)

age_days = -np.random.randint(18 * 365, 70 * 365, size=N)  # DAYS_BIRTH (negative)
# employment duration: pensioners get a large positive "365243" sentinel (as in real dataset)
employed_days = np.where(
    income_type == "Pensioner",
    365243,
    -np.random.randint(0, 20 * 365, size=N)
)

mobile = np.ones(N, dtype=int)
work_phone = np.random.choice([0, 1], size=N, p=[0.75, 0.25])
phone = np.random.choice([0, 1], size=N, p=[0.6, 0.4])
email = np.random.choice([0, 1], size=N, p=[0.85, 0.15])

occupation = np.random.choice(
    ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers",
     "High skill tech staff", "Accountants", "Medicine staff", np.nan],
    size=N, p=[0.2, 0.15, 0.13, 0.12, 0.1, 0.08, 0.07, 0.07, 0.08]
)

family_members = (children + np.random.choice([1, 2], size=N)).clip(1, 7)

application_record = pd.DataFrame({
    "ID": ids,
    "CODE_GENDER": gender,
    "FLAG_OWN_CAR": own_car,
    "FLAG_OWN_REALTY": own_realty,
    "CNT_CHILDREN": children,
    "AMT_INCOME_TOTAL": annual_income,
    "NAME_INCOME_TYPE": income_type,
    "NAME_EDUCATION_TYPE": education,
    "NAME_FAMILY_STATUS": family_status,
    "NAME_HOUSING_TYPE": housing_type,
    "DAYS_BIRTH": age_days,
    "DAYS_EMPLOYED": employed_days,
    "FLAG_MOBIL": mobile,
    "FLAG_WORK_PHONE": work_phone,
    "FLAG_PHONE": phone,
    "FLAG_EMAIL": email,
    "OCCUPATION_TYPE": occupation,
    "CNT_FAM_MEMBERS": family_members,
})

# ---------------------------------------------------------------------
# credit_record.csv -> monthly payment status history per applicant
# STATUS: 0 = 1-29 days past due, 1 = 30-59, 2 = 60-89, 3 = 90-119,
#         4 = 120-149, 5 = overdue/bad debt, C = paid off that month, X = no loan
# ---------------------------------------------------------------------
records = []
# risk score per applicant drives how likely they are to go delinquent
risk_score = (
    0.35 * (annual_income < np.percentile(annual_income, 30)).astype(float)
    + 0.25 * (income_type == "Student").astype(float)
    + 0.20 * (employed_days > 0).astype(float) * 0  # pensioners not inherently risky
    + 0.15 * (family_members > 4).astype(float)
    + np.random.uniform(0, 0.3, size=N)
)

for i, appid in enumerate(ids):
    months = np.random.randint(6, 25)
    r = risk_score[i]
    for m in range(months):
        month_balance = -m
        roll = np.random.random()
        if roll < r * 0.15:
            status = np.random.choice(["2", "3", "4", "5"], p=[0.4, 0.3, 0.2, 0.1])
        elif roll < r * 0.4:
            status = np.random.choice(["0", "1"], p=[0.7, 0.3])
        elif roll < 0.5:
            status = "C"
        else:
            status = "X"
        records.append((appid, month_balance, status))

credit_record = pd.DataFrame(records, columns=["ID", "MONTHS_BALANCE", "STATUS"])

application_record.to_csv("application_record.csv", index=False)
credit_record.to_csv("credit_record.csv", index=False)

print("Generated application_record.csv:", application_record.shape)
print("Generated credit_record.csv:", credit_record.shape)
