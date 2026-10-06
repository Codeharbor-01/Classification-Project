from datetime import datetime
from html import escape
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT_DIRECTORY = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIRECTORY / "Dataset" / "Cleaned_dataset.csv"
SCORES_DIRECTORY = ROOT_DIRECTORY / "Scores"
MODELS_DIRECTORY = ROOT_DIRECTORY / "Pkl_Files"

MODEL_FILES = {
    "Random Forest": ("RandomForestClassifier.pkl", "RandomForestClassifier_scores.pkl"),
    "Logistic Regression": ("LogisticRegression.pkl", "LogisticRegression_scores.pkl"),
    "K-Nearest Neighbors": ("KNNClassifier.pkl", "KNNClassifier_scores.pkl"),
}
METRICS = ("accuracy", "precision", "recall", "f1")

NUMERIC_FIELDS = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Waist_Circumference_cm",
    "Blood_Glucose",
    "HbA1c",
    "Fasting_Blood_Sugar",
    "Insulin_Level",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Total_Cholesterol",
    "HDL",
    "LDL",
    "Triglycerides",
    "Heart_Rate",
    "Exercise_Hours_Per_Week",
    "Daily_Walking_Minutes",
    "Sleep_Hours",
    "Daily_Water_Intake_L",
]
CATEGORICAL_FIELDS = [
    "Gender",
    "Country",
    "Physical_Activity_Level",
    "Diet_Quality",
    "Sugar_Intake_Level",
    "Stress_Level",
    "Smoking_Status",
    "Alcohol_Consumption",
    "Medication_Adherence",
    "Work_Type",
    "Residence_Type",
]
BOOLEAN_FIELDS = [
    "Family_History_Diabetes",
    "Hypertension",
    "Heart_Disease",
    "Fatty_Liver",
    "PCOS",
]

FIELD_LABELS = {
    "Age": "Age (years)",
    "Height_cm": "Height (cm)",
    "Weight_kg": "Weight (kg)",
    "BMI": "BMI",
    "Waist_Circumference_cm": "Waist circumference (cm)",
    "Blood_Glucose": "Blood glucose (mg/dL)",
    "HbA1c": "HbA1c (%)",
    "Fasting_Blood_Sugar": "Fasting blood sugar (mg/dL)",
    "Insulin_Level": "Insulin level",
    "Blood_Pressure_Systolic": "Systolic blood pressure (mmHg)",
    "Blood_Pressure_Diastolic": "Diastolic blood pressure (mmHg)",
    "Total_Cholesterol": "Total cholesterol (mg/dL)",
    "HDL": "HDL cholesterol (mg/dL)",
    "LDL": "LDL cholesterol (mg/dL)",
    "Triglycerides": "Triglycerides (mg/dL)",
    "Heart_Rate": "Heart rate (bpm)",
    "Exercise_Hours_Per_Week": "Exercise (hours/week)",
    "Daily_Walking_Minutes": "Walking (minutes/day)",
    "Sleep_Hours": "Sleep (hours/day)",
    "Daily_Water_Intake_L": "Water intake (L/day)",
    "Family_History_Diabetes": "Family history of diabetes",
    "Heart_Disease": "History of heart disease",
    "Fatty_Liver": "Fatty liver",
}

ORDINAL_OPTIONS = {
    "Physical_Activity_Level": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Average", "Healthy"],
    "Sugar_Intake_Level": ["Low", "Moderate", "High"],
    "Stress_Level": ["Low", "Moderate", "High"],
    "Alcohol_Consumption": ["Never", "Occasionally", "Frequently"],
    "Medication_Adherence": ["Poor", "Average", "Good"],
}


def field_label(field):
    return FIELD_LABELS.get(field, field.replace("_", " "))


if not DATA_PATH.is_file():
    st.error(f"Training data not found: {DATA_PATH}")
    st.stop()

training_data = pd.read_csv(DATA_PATH)
training_data = training_data.drop(columns=["Unnamed: 0"], errors="ignore")
required_fields = set(NUMERIC_FIELDS + CATEGORICAL_FIELDS + BOOLEAN_FIELDS)
missing_fields = sorted(required_fields.difference(training_data.columns))
if missing_fields:
    st.error(f"Training data is missing required prediction fields: {', '.join(missing_fields)}")
    st.stop()

model_scores = {}
for model_name, (_, score_filename) in MODEL_FILES.items():
    score_path = SCORES_DIRECTORY / score_filename
    if not score_path.is_file():
        st.error(f"Evaluation scores not found: {score_path}")
        st.stop()

    score_data = joblib.load(score_path)
    if not isinstance(score_data, dict) or any(metric not in score_data for metric in METRICS):
        st.error(f"Invalid evaluation scores in {score_path}")
        st.stop()

    try:
        metric_values = {metric: float(score_data[metric]) for metric in METRICS}
    except (TypeError, ValueError):
        st.error(f"Invalid metric value in {score_path}")
        st.stop()
    if any(not 0 <= value <= 1 for value in metric_values.values()):
        st.error(f"Expected all evaluation scores in {score_path} to be between 0 and 1")
        st.stop()
    model_scores[model_name] = metric_values

best_model_name = max(
    model_scores,
    key=lambda name: sum(model_scores[name].values()) / len(METRICS),
)
best_model_path = MODELS_DIRECTORY / MODEL_FILES[best_model_name][0]
if not best_model_path.is_file():
    st.error(f"Saved model not found: {best_model_path}")
    st.stop()


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


model = load_model(str(best_model_path))
model_features = list(getattr(model, "feature_names_in_", []))
expected_input_fields = set(NUMERIC_FIELDS + CATEGORICAL_FIELDS + BOOLEAN_FIELDS)
missing_model_fields = sorted(expected_input_fields.difference(model_features))
if missing_model_fields:
    st.error(
        "The selected model does not accept all form fields: "
        + ", ".join(missing_model_fields)
    )
    st.stop()

st.title("Diabetes Risk Prediction")
st.write(
    "Enter the available measurements and health information to generate a "
    "risk-group prediction. Fields start with typical values from the project "
    "dataset; adjust them to match the person being assessed."
)

best_scores = model_scores[best_model_name]
average_score = sum(best_scores.values()) / len(METRICS)
st.info(
    f"Selected model: **{best_model_name}** — highest average across the saved "
    f"accuracy, precision, recall, and F1 scores ({average_score:.1%})."
)
st.caption(
    "Predictions are experimental and for this project only. They are not a "
    "diagnosis, medical advice, or a substitute for a qualified health professional."
)
if "Unnamed: 0" in model_features:
    st.caption(
        "Model note: the saved pipeline includes the training CSV's row index. "
        "For a new record, the training-set median index is used as a neutral value."
    )

form_values = {}


def add_numeric_fields(fields):
    columns = st.columns(2)
    for index, field in enumerate(fields):
        values = pd.to_numeric(training_data[field], errors="coerce").dropna()
        minimum = float(values.min())
        maximum = float(values.max())
        default = min(max(float(values.median()), minimum), maximum)
        integer_like = (values % 1 == 0).all()
        with columns[index % 2]:
            form_values[field] = st.number_input(
                field_label(field),
                min_value=minimum,
                max_value=maximum,
                value=default,
                step=1.0 if integer_like else 0.1,
                format="%.1f",
                key=f"prediction_{field}",
            )


def add_categorical_fields(fields):
    columns = st.columns(2)
    for index, field in enumerate(fields):
        if field in ORDINAL_OPTIONS:
            options = ORDINAL_OPTIONS[field]
        else:
            options = sorted(training_data[field].dropna().unique().tolist())
        if not options:
            st.error(f"No valid choices are available for {field_label(field)}.")
            st.stop()
        mode = training_data[field].mode(dropna=True)
        default = mode.iloc[0] if not mode.empty and mode.iloc[0] in options else options[0]
        with columns[index % 2]:
            form_values[field] = st.selectbox(
                field_label(field),
                options=options,
                index=options.index(default),
                key=f"prediction_{field}",
            )


def add_boolean_fields(fields):
    columns = st.columns(2)
    for index, field in enumerate(fields):
        mode = training_data[field].mode(dropna=True)
        default = bool(mode.iloc[0]) if not mode.empty else False
        with columns[index % 2]:
            form_values[field] = st.checkbox(
                field_label(field),
                value=default,
                key=f"prediction_{field}",
            )


with st.form("diabetes_risk_prediction_form"):
    st.subheader("Personal details")
    add_numeric_fields(["Age", "Height_cm", "Weight_kg", "BMI", "Waist_Circumference_cm"])
    add_categorical_fields(["Gender", "Country"])

    st.divider()
    st.subheader("Health measurements")
    add_numeric_fields(
        [
            "Blood_Glucose",
            "HbA1c",
            "Fasting_Blood_Sugar",
            "Insulin_Level",
            "Blood_Pressure_Systolic",
            "Blood_Pressure_Diastolic",
            "Total_Cholesterol",
            "HDL",
            "LDL",
            "Triglycerides",
            "Heart_Rate",
        ]
    )

    st.divider()
    st.subheader("Lifestyle")
    add_categorical_fields(
        [
            "Physical_Activity_Level",
            "Diet_Quality",
            "Sugar_Intake_Level",
            "Stress_Level",
            "Smoking_Status",
            "Alcohol_Consumption",
            "Medication_Adherence",
            "Work_Type",
            "Residence_Type",
        ]
    )
    add_numeric_fields(
        [
            "Exercise_Hours_Per_Week",
            "Daily_Walking_Minutes",
            "Sleep_Hours",
            "Daily_Water_Intake_L",
        ]
    )

    st.divider()
    st.subheader("Health history")
    add_boolean_fields(BOOLEAN_FIELDS)

    submitted = st.form_submit_button("Generate prediction report", type="primary")

if submitted:
    prediction_data = pd.DataFrame([form_values])

    # This saved pipeline was trained with the CSV's row-index column.
    # Use its training median for new records because it is not a user feature.
    if "Unnamed: 0" in model_features:
        original_data = pd.read_csv(DATA_PATH, usecols=["Unnamed: 0"])
        prediction_data["Unnamed: 0"] = float(original_data["Unnamed: 0"].median())

    unhandled_features = sorted(set(model_features).difference(prediction_data.columns))
    if unhandled_features:
        st.error("Unable to prepare model inputs: " + ", ".join(unhandled_features))
        st.stop()
    prediction_data = prediction_data.reindex(columns=model_features)
    prediction = model.predict(prediction_data)[0]

    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    risk_group = escape(str(prediction))
    report_sections = {
        "Personal details": [
            "Age",
            "Gender",
            "Country",
            "Height_cm",
            "Weight_kg",
            "BMI",
            "Waist_Circumference_cm",
        ],
        "Health measurements": [
            "Blood_Glucose",
            "HbA1c",
            "Fasting_Blood_Sugar",
            "Insulin_Level",
            "Blood_Pressure_Systolic",
            "Blood_Pressure_Diastolic",
            "Total_Cholesterol",
            "HDL",
            "LDL",
            "Triglycerides",
            "Heart_Rate",
        ],
        "Lifestyle": [
            "Physical_Activity_Level",
            "Exercise_Hours_Per_Week",
            "Daily_Walking_Minutes",
            "Diet_Quality",
            "Sugar_Intake_Level",
            "Sleep_Hours",
            "Stress_Level",
            "Smoking_Status",
            "Alcohol_Consumption",
            "Medication_Adherence",
            "Work_Type",
            "Residence_Type",
            "Daily_Water_Intake_L",
        ],
        "Health history": BOOLEAN_FIELDS,
    }
    details_html = []
    for section, fields in report_sections.items():
        rows = []
        for field in fields:
            value = form_values[field]
            if field in BOOLEAN_FIELDS:
                display_value = "Yes" if value else "No"
            elif field in NUMERIC_FIELDS:
                display_value = f"{value:g}"
            else:
                display_value = str(value)
            rows.append(
                f"<tr><th scope=\"row\">{escape(field_label(field))}</th>"
                f"<td>{escape(display_value)}</td></tr>"
            )
        details_html.append(
            f'<section class="report-section"><h3>{escape(section)}</h3>'
            f'<table class="report-details"><tbody>{"".join(rows)}</tbody></table></section>'
        )
    details_html = "".join(details_html)
    report_html = f"""
    <style>
        .prediction-report {{
            max-width: 780px;
            margin: 1.5rem auto;
            padding: 2.5rem;
            border: 1px solid #d8e1e5;
            border-top: 6px solid #176b62;
            border-radius: 12px;
            background: #ffffff;
            color: #20343b;
            box-shadow: 0 8px 24px rgba(16, 44, 56, 0.08);
            font-family: Arial, sans-serif;
        }}
        .prediction-report .report-kicker {{
            margin: 0 0 0.35rem;
            color: #527078;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }}
        .prediction-report h2 {{
            margin: 0;
            color: #173c45;
            font-size: 1.7rem;
        }}
        .prediction-report .report-date {{
            margin-top: 0.55rem;
            color: #63777d;
            font-size: 0.9rem;
        }}
        .prediction-report .report-rule {{
            margin: 1.5rem 0;
            border: 0;
            border-top: 1px solid #dce5e8;
        }}
        .prediction-report .result-label {{
            margin: 0 0 0.65rem;
            color: #527078;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}
        .prediction-report .result {{
            margin: 0;
            color: #176b62;
            font-size: 2.25rem;
            font-weight: 700;
        }}
        .prediction-report .report-disclaimer {{
            margin: 1.75rem 0 0;
            padding: 0.9rem 1rem;
            border-left: 3px solid #d4a449;
            background: #fbf7ed;
            color: #665a40;
            font-size: 0.88rem;
            line-height: 1.55;
        }}
        .prediction-report .report-section {{
            margin-top: 1.6rem;
        }}
        .prediction-report .report-section h3 {{
            margin: 0 0 0.55rem;
            padding-bottom: 0.45rem;
            border-bottom: 1px solid #dce5e8;
            color: #25535b;
            font-size: 1.05rem;
        }}
        .prediction-report .report-details {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.92rem;
        }}
        .prediction-report .report-details th,
        .prediction-report .report-details td {{
            width: 50%;
            padding: 0.45rem 0.65rem;
            border-bottom: 1px solid #edf1f2;
            text-align: left;
            vertical-align: top;
        }}
        .prediction-report .report-details th {{
            color: #527078;
            font-weight: 600;
        }}
        .prediction-report .report-details td {{
            color: #20343b;
        }}
        @media print {{
            .prediction-report {{
                max-width: none;
                margin: 0;
                border: 1px solid #ccd5d9;
                border-top: 5px solid #176b62;
                box-shadow: none;
            }}
        }}
    </style>
    <section class="prediction-report">
        <p class="report-kicker">Automated screening result</p>
        <h2>Diabetes Risk Report</h2>
        <p class="report-date">Generated {escape(generated_at)}</p>
        <hr class="report-rule">
        <p class="result-label">Predicted risk group</p>
        <p class="result">{risk_group}</p>
        {details_html}
        <p class="report-disclaimer">
            This is an automated prediction from a student project, not a
            physician's report, medical diagnosis, or substitute for professional
            medical advice.
        </p>
    </section>
    """

    st.divider()
    st.markdown(report_html, unsafe_allow_html=True)
    st.download_button(
        "Download report",
        data=f"<!doctype html><html><head><meta charset=\"utf-8\"><title>Diabetes Risk Report</title></head><body>{report_html}</body></html>",
        file_name="diabetes_risk_report.html",
        mime="text/html",
    )