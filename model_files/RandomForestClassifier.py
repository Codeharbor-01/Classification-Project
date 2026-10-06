import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from category_encoders import BinaryEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV

df = pd.read_csv(r'Dataset/Cleaned_dataset.csv')

x = df.drop(columns=['Patient_ID','AI_Health_Recommendation','Doctor_Consultation_Needed','Diabetes_Risk_Score','Diabetes_Risk'])
y = df['Diabetes_Risk']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

num_cols = [
    'Age',
    'Height_cm',
    'Weight_kg',
    'BMI',
    'Waist_Circumference_cm',
    'Blood_Glucose',
    'HbA1c',
    'Fasting_Blood_Sugar',
    'Insulin_Level',
    'Blood_Pressure_Systolic',
    'Blood_Pressure_Diastolic',
    'Total_Cholesterol',
    'HDL',
    'LDL',
    'Triglycerides',
    'Heart_Rate',
    'Exercise_Hours_Per_Week',
    'Daily_Walking_Minutes',
    'Sleep_Hours',
    'Daily_Water_Intake_L',
]

ord_cols = [
    'Physical_Activity_Level',
    'Diet_Quality',
    'Sugar_Intake_Level',
    'Stress_Level',
    'Alcohol_Consumption',
    'Medication_Adherence'
]

physical_order = ['Low', 'Moderate', 'High']
diet_order = ['Poor', 'Average', 'Healthy']
sugar_order = ['Low', 'Moderate', 'High']
stress_order = ['Low', 'Moderate', 'High']
alcohol_order = ['Never', 'Occasionally', 'Frequently']
medication_order = ['Poor', 'Average', 'Good']

nom_cols =[
    'Gender',
    'Country',
    'Smoking_Status',
    'Work_Type',
    'Residence_Type'
]
bool_cols = [
    'Family_History_Diabetes',
    'Hypertension',
    'Heart_Disease',
    'Fatty_Liver',
    'PCOS',
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num_scaled', StandardScaler(), num_cols),
        ('ord_encoded', OrdinalEncoder(categories=[physical_order, diet_order, sugar_order, stress_order, alcohol_order, medication_order]), ord_cols),
        ('nom_encoded', BinaryEncoder(), nom_cols)
    ],remainder='passthrough'
)

pipe = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ]
)

params = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2]
}

grid_model = GridSearchCV(pipe, params, cv=5, scoring='accuracy', n_jobs=-1)
grid_model.fit(x_train, y_train)

best_model = grid_model.best_estimator_

y_pred = best_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

scores = {
    "accuracy": accuracy,
    "precision": precision,
    "f1": f1,
    "recall": recall
}

joblib.dump(best_model, 'Pkl_Files/RandomForestClassifier.pkl')
joblib.dump(scores, 'Scores/RandomForestClassifier_scores.pkl')