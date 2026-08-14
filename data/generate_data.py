"""
Generates a synthetic student performance dataset that follows the same
structure and realistic feature relationships as the well-known UCI
"Student Performance" dataset (Cortez & Silva, 2008).

Why synthetic: guarantees a clean, ready-to-use dataset with no download
dependency, while preserving realistic correlations (more study time and
fewer absences/failures push grades up; alcohol consumption and travel
time push grades down; parental education has a mild positive effect).
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 900  # number of students

schools = np.random.choice(["GP", "MS"], N, p=[0.7, 0.3])
sex = np.random.choice(["F", "M"], N, p=[0.52, 0.48])
age = np.random.randint(15, 20, N)
address = np.random.choice(["U", "R"], N, p=[0.68, 0.32])
famsize = np.random.choice(["LE3", "GT3"], N, p=[0.4, 0.6])
Pstatus = np.random.choice(["T", "A"], N, p=[0.9, 0.1])
Medu = np.random.randint(0, 5, N)          # mother's education 0-4
Fedu = np.random.randint(0, 5, N)          # father's education 0-4
studytime = np.random.randint(1, 5, N)      # weekly study time 1-4
failures = np.random.choice([0, 1, 2, 3], N, p=[0.72, 0.15, 0.08, 0.05])
schoolsup = np.random.choice(["yes", "no"], N, p=[0.12, 0.88])
famsup = np.random.choice(["yes", "no"], N, p=[0.6, 0.4])
paid = np.random.choice(["yes", "no"], N, p=[0.4, 0.6])
activities = np.random.choice(["yes", "no"], N, p=[0.5, 0.5])
higher = np.random.choice(["yes", "no"], N, p=[0.9, 0.1])
internet = np.random.choice(["yes", "no"], N, p=[0.83, 0.17])
romantic = np.random.choice(["yes", "no"], N, p=[0.35, 0.65])
famrel = np.random.randint(1, 6, N)
freetime = np.random.randint(1, 6, N)
goout = np.random.randint(1, 6, N)
Dalc = np.random.randint(1, 6, N)           # workday alcohol
Walc = np.random.randint(1, 6, N)           # weekend alcohol
health = np.random.randint(1, 6, N)
absences = np.random.poisson(4, N).clip(0, 30)
traveltime = np.random.randint(1, 5, N)

# --- Build grades with realistic underlying relationships + noise ---
base = 11.0
score = (
    base
    + 1.3 * studytime
    - 1.8 * failures
    - 0.12 * absences
    - 0.35 * traveltime
    + 0.25 * (Medu + Fedu)
    - 0.3 * Dalc
    - 0.2 * Walc
    + 0.15 * famrel
    + 0.1 * health
    + np.where(schoolsup == "yes", 0.6, 0)
    + np.where(internet == "yes", 0.4, 0)
    + np.where(higher == "yes", 0.8, 0)
    + np.random.normal(0, 2.2, N)
)

G1 = np.clip(np.round(score + np.random.normal(0, 1.2, N)), 0, 20).astype(int)
G2 = np.clip(np.round(0.6 * G1 + 0.4 * score + np.random.normal(0, 1.0, N)), 0, 20).astype(int)
G3 = np.clip(np.round(0.5 * G2 + 0.3 * G1 + 0.2 * score + np.random.normal(0, 0.8, N)), 0, 20).astype(int)

df = pd.DataFrame({
    "school": schools, "sex": sex, "age": age, "address": address,
    "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
    "traveltime": traveltime, "studytime": studytime, "failures": failures,
    "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
    "activities": activities, "higher": higher, "internet": internet,
    "romantic": romantic, "famrel": famrel, "freetime": freetime,
    "goout": goout, "Dalc": Dalc, "Walc": Walc, "health": health,
    "absences": absences, "G1": G1, "G2": G2, "G3": G3,
})

df["pass"] = (df["G3"] >= 10).astype(int)  # binary target: pass/fail (out of 20)

df.to_csv("data/student_performance.csv", index=False)
print(df.shape)
print(df.head())
print("\nPass rate:", df["pass"].mean().round(3))
