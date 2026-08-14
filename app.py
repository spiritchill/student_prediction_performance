import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    reg_model = joblib.load("models/best_regressor.pkl")
    clf_model = joblib.load("models/best_classifier.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")

    return reg_model, clf_model, scaler, encoders, feature_cols


reg_model, clf_model, scaler, encoders, feature_cols = load_artifacts()


st.title("🎓 AI-Driven Student Performance Predictor")

st.caption(
    "Predicts a student's final grade (G3, out of 20) "
    "and pass/fail outcome from academic & lifestyle factors."
)

st.divider()

col1, col2 = st.columns(2)


with col1:
    school = st.selectbox("School", ["GP", "MS"])

    sex = st.selectbox("Sex", ["F", "M"])

    age = st.slider("Age", 15, 19, 17)

    address = st.selectbox(
        "Address",
        ["U", "R"],
        format_func=lambda x: "Urban" if x == "U" else "Rural"
    )

    famsize = st.selectbox(
        "Family Size",
        ["LE3", "GT3"],
        format_func=lambda x: "≤3" if x == "LE3" else ">3"
    )

    Pstatus = st.selectbox(
        "Parents' Cohabitation",
        ["T", "A"],
        format_func=lambda x: "Together" if x == "T" else "Apart"
    )

    Medu = st.slider(
        "Mother's Education (0=none, 4=higher)",
        0, 4, 2
    )

    Fedu = st.slider(
        "Father's Education (0=none, 4=higher)",
        0, 4, 2
    )

    traveltime = st.slider(
        "Travel Time to School (1=short, 4=long)",
        1, 4, 1
    )

    studytime = st.slider(
        "Weekly Study Time (1=low, 4=high)",
        1, 4, 2
    )

    failures = st.slider(
        "Past Class Failures",
        0, 3, 0
    )

    schoolsup = st.selectbox(
        "Extra School Support",
        ["yes", "no"]
    )

    famsup = st.selectbox(
        "Family Educational Support",
        ["yes", "no"]
    )

    paid = st.selectbox(
        "Extra Paid Classes",
        ["yes", "no"]
    )

    activities = st.selectbox(
        "Extracurricular Activities",
        ["yes", "no"]
    )


with col2:
    higher = st.selectbox(
        "Wants Higher Education",
        ["yes", "no"]
    )

    internet = st.selectbox(
        "Internet Access at Home",
        ["yes", "no"]
    )

    romantic = st.selectbox(
        "In a Relationship",
        ["yes", "no"]
    )

    famrel = st.slider(
        "Family Relationship Quality (1-5)",
        1, 5, 4
    )

    freetime = st.slider(
        "Free Time After School (1-5)",
        1, 5, 3
    )

    goout = st.slider(
        "Going Out with Friends (1-5)",
        1, 5, 3
    )

    Dalc = st.slider(
        "Workday Alcohol Consumption (1-5)",
        1, 5, 1
    )

    Walc = st.slider(
        "Weekend Alcohol Consumption (1-5)",
        1, 5, 1
    )

    health = st.slider(
        "Current Health Status (1-5)",
        1, 5, 4
    )

    absences = st.slider(
        "Number of Absences",
        0, 30, 4
    )

    G1 = st.slider(
        "First Period Grade (G1)",
        0, 20, 12
    )

    G2 = st.slider(
        "Second Period Grade (G2)",
        0, 20, 12
    )


st.divider()


if st.button(
    "Predict Performance",
    type="primary",
    use_container_width=True
):

    raw = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences,
        "G1": G1,
        "G2": G2,
    }

    row = pd.DataFrame([raw])

    # Encode categorical columns
    for col, le in encoders.items():
        if col in row.columns:
            row[col] = le.transform(row[col])

    # Keep the exact feature order used during training
    row = row[feature_cols]

    # Regression prediction
    # Linear Regression was trained on scaled features
    predicted_grade = float(
        reg_model.predict(scaler.transform(row))[0]
    )

    # Keep grade between 0 and 20
    predicted_grade = max(
        0,
        min(20, predicted_grade)
    )

    # Classification prediction
    # Random Forest was trained on unscaled features
    pass_pred = clf_model.predict(row)[0]

    pass_proba = (
        clf_model.predict_proba(row)[0][1]
        if hasattr(clf_model, "predict_proba")
        else None
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Predicted Final Grade (G3)",
            f"{predicted_grade:.1f} / 20"
        )

    with c2:
        st.metric(
            "Predicted Outcome",
            "✅ Pass" if pass_pred == 1 else "❌ Fail",
            delta=(
                f"{pass_proba * 100:.1f}% confidence"
                if pass_proba is not None
                else None
            )
        )

    if predicted_grade < 10:
        st.warning(
            "This student profile is at risk of underperforming. "
            "Consider recommending extra support, tutoring, "
            "or reduced absences."
        )
    else:
        st.success(
            "This student profile suggests solid academic performance."
        )


st.divider()

with st.expander("ℹ️ About this project"):
    st.write("""
    This tool uses two machine learning models trained on the
    UCI Student Performance dataset (student-mat.csv):

    - **Regression model:** Linear Regression predicts the exact
      final grade (G3, 0–20).

    - **Classification model:** Random Forest predicts a binary
      Pass/Fail outcome.

    The regression model was evaluated using RMSE, MAE, and R².

    The classification model was evaluated using Accuracy,
    Precision, Recall, and F1-score.

    The models use demographic, family, lifestyle, and academic
    factors including G1 and G2.
    """)