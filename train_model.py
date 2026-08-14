"""
AI-Driven Student Performance Prediction System
Trains models to (a) predict final grade G3 (regression) and
(b) predict pass/fail (classification), evaluates them properly,
and saves the best models + plots for the report/app.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
sns.set_style("whitegrid")

df = pd.read_csv("data/student_performance.csv")

# ------------------------------------------------------------------
# 1. EDA — save a few key plots for the report
# ------------------------------------------------------------------
plt.figure(figsize=(7, 4))
sns.histplot(df["G3"], bins=20, kde=True, color="#1F3864")
plt.title("Distribution of Final Grade (G3)")
plt.xlabel("Final Grade (0-20)")
plt.tight_layout()
plt.savefig("plots/grade_distribution.png", dpi=130)
plt.close()

plt.figure(figsize=(6, 5))
corr_cols = ["studytime", "failures", "absences", "Medu", "Fedu", "Dalc", "Walc", "goout", "G1", "G2", "G3"]
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Feature Correlation with Final Grade")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=130)
plt.close()

plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="studytime", y="G3", palette="Blues")
plt.title("Study Time vs Final Grade")
plt.xlabel("Weekly Study Time (1=low, 4=high)")
plt.ylabel("Final Grade")
plt.tight_layout()
plt.savefig("plots/studytime_vs_grade.png", dpi=130)
plt.close()

plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x="absences", y="G3", hue="pass", palette=["#C0392B", "#1F3864"], alpha=0.6)
plt.title("Absences vs Final Grade")
plt.tight_layout()
plt.savefig("plots/absences_vs_grade.png", dpi=130)
plt.close()

# ------------------------------------------------------------------
# 2. Preprocessing
# ------------------------------------------------------------------
data = df.copy()
cat_cols = data.select_dtypes(include="object").columns.tolist()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

joblib.dump(encoders, "models/encoders.pkl")

feature_cols = [c for c in data.columns if c not in ["G3", "pass"]]
X = data[feature_cols]
y_reg = data["G3"]
y_clf = data["pass"]

X_train, X_test, yreg_train, yreg_test, yclf_train, yclf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")

# ------------------------------------------------------------------
# 3. Regression: predict final grade (G3)
# ------------------------------------------------------------------
lr = LinearRegression().fit(X_train_s, yreg_train)
rf_reg = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42).fit(X_train, yreg_train)

lr_pred = lr.predict(X_test_s)
rf_pred = rf_reg.predict(X_test)

reg_results = {
    "Linear Regression": {
        "RMSE": float(np.sqrt(mean_squared_error(yreg_test, lr_pred))),
        "MAE": float(mean_absolute_error(yreg_test, lr_pred)),
        "R2": float(r2_score(yreg_test, lr_pred)),
    },
    "Random Forest Regressor": {
        "RMSE": float(np.sqrt(mean_squared_error(yreg_test, rf_pred))),
        "MAE": float(mean_absolute_error(yreg_test, rf_pred)),
        "R2": float(r2_score(yreg_test, rf_pred)),
    },
}

best_reg_name = min(reg_results, key=lambda k: reg_results[k]["RMSE"])
best_reg_model = rf_reg if best_reg_name == "Random Forest Regressor" else lr
joblib.dump(best_reg_model, "models/best_regressor.pkl")

plt.figure(figsize=(6, 6))
plt.scatter(yreg_test, rf_pred, alpha=0.5, color="#1F3864")
plt.plot([0, 20], [0, 20], "r--")
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title(f"Actual vs Predicted Final Grade ({best_reg_name})")
plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png", dpi=130)
plt.close()

# ------------------------------------------------------------------
# 4. Classification: predict pass/fail
# ------------------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000).fit(X_train_s, yclf_train)
rf_clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=42).fit(X_train, yclf_train)

log_pred = log_reg.predict(X_test_s)
rf_clf_pred = rf_clf.predict(X_test)

def clf_metrics(y_true, y_pred):
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

clf_results = {
    "Logistic Regression": clf_metrics(yclf_test, log_pred),
    "Random Forest Classifier": clf_metrics(yclf_test, rf_clf_pred),
}

best_clf_name = max(clf_results, key=lambda k: clf_results[k]["F1"])
best_clf_model = rf_clf if best_clf_name == "Random Forest Classifier" else log_reg
joblib.dump(best_clf_model, "models/best_classifier.pkl")

cm = confusion_matrix(yclf_test, rf_clf_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, cmap="Blues", colorbar=False)
plt.title(f"Confusion Matrix ({best_clf_name})")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=130)
plt.close()

# Feature importance (Random Forest regressor)
importances = pd.Series(rf_reg.feature_importances_, index=feature_cols).sort_values(ascending=False).head(10)
plt.figure(figsize=(7, 5))
sns.barplot(x=importances.values, y=importances.index, color="#1F3864")
plt.title("Top 10 Feature Importances (Predicting Final Grade)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=130)
plt.close()

# ------------------------------------------------------------------
# 5. Save results summary
# ------------------------------------------------------------------
results = {
    "regression": reg_results,
    "best_regressor": best_reg_name,
    "classification": clf_results,
    "best_classifier": best_clf_name,
}
with open("models/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("\nAll models and plots saved.")
