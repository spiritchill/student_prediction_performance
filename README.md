# AI-Driven Student Performance Prediction System

Predicts a student's final academic grade and pass/fail outcome from demographic,
family, and lifestyle factors, using machine learning.

## Problem Statement
Educational institutions want to identify students at risk of underperforming
early enough to intervene (extra support, tutoring, counseling). This project
builds ML models that predict:
1. **Final grade (G3)** — a regression problem (0–20 scale)
2. **Pass / Fail outcome** — a binary classification problem

## Dataset
900 student records with 26 features covering demographics (age, sex, address),
family background (parents' education, family size), study habits (weekly study
time, past failures, absences), and lifestyle factors (alcohol consumption,
free time, going out, health), plus period grades G1/G2 and final grade G3.
The schema mirrors the well-known UCI "Student Performance" dataset
(Cortez & Silva, 2008), generated with realistic, documented correlations
(e.g., more study time and fewer failures/absences raise grades; higher
alcohol consumption and travel time lower them).

##Working Site
https://studentpredictionperformancenew.streamlit.app/

## Approach
1. **EDA** — explored grade distribution, correlations between lifestyle/academic
   features and final grade, and the relationship between study time / absences
   and performance (see `plots/`).
2. **Preprocessing** — label-encoded categorical variables, scaled numeric
   features for linear models, and used an 80/20 train-test split (stratified
   on pass/fail).
3. **Regression models** — Linear Regression and Random Forest Regressor to
   predict the exact G3 score. Evaluated with **RMSE, MAE, and R²** (not just
   a single metric).
4. **Classification models** — Logistic Regression and Random Forest Classifier
   to predict Pass/Fail. Evaluated with **Accuracy, Precision, Recall, and F1**,
   plus a confusion matrix, since accuracy alone can be misleading if the
   pass/fail split is uneven.
5. **Feature importance** — used Random Forest to identify which factors most
   influence final grade (see `plots/feature_importance.png`).
6. **Deployment** — built an interactive Streamlit app where a user inputs a
   student profile and receives a predicted grade and pass/fail outcome.

## Results

| Regression Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | ~0.86 | ~0.68 | ~0.93 |
| Random Forest Regressor | ~1.00 | ~0.79 | ~0.91 |

| Classification Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | ~96.1% | ~96.3% | ~99.4% | ~97.8% |
| Random Forest Classifier | ~98.3% | ~98.1% | ~100% | ~99.1% |

(Full numbers saved in `models/results.json`.)

**Key insight from feature importance:** prior period grades (G1, G2) and
number of past failures are the strongest predictors of final grade, followed
by weekly study time — consistent with educational research on academic
performance.

## Tech Stack
- Python, Pandas, NumPy — data handling
- Scikit-learn — modeling & evaluation
- Matplotlib, Seaborn — visualization
- Streamlit — interactive demo app
- Joblib — model persistence

## How to Run

```bash
pip install -r requirements.txt
python data/generate_data.py     # generates the dataset
python train_model.py            # trains models, saves plots + models
streamlit run app.py             # launches the interactive app
```

## Project Structure
```
student-performance/
├── data/
│   ├── generate_data.py
│   └── student_performance.csv
├── models/                # saved trained models + encoders + results.json
├── plots/                 # EDA and evaluation visualizations
├── train_model.py         # full training & evaluation pipeline
├── app.py                 # Streamlit demo app
├── requirements.txt
└── README.md
```

## Future Improvements
- Hyperparameter tuning (GridSearchCV) for Random Forest
- Try gradient boosting (XGBoost/LightGBM) for comparison
- SHAP values for per-prediction explainability
- Deploy publicly (Streamlit Community Cloud) for live access
