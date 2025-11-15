import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import joblib

np.random.seed(42)

n = 2000
age = np.random.randint(0, 100, n)
gender = np.random.choice(["Male", "Female", "Other"], n, p=[0.48, 0.48, 0.04])
appt_type = np.random.choice(["General Checkup", "Specialist", "Follow-up"], n, p=[0.5, 0.3, 0.2])
days_until = np.random.randint(0, 31, n)
prev_no_shows = np.random.poisson(0.6, n)
prev_no_shows = np.clip(prev_no_shows, 0, 5)
sms_sent = np.random.choice(["Yes", "No"], n, p=[0.8, 0.2])

base_prob = (
    0.02 * (age < 18).astype(float) +
    0.01 * (age > 70).astype(float) +
    0.03 * (appt_type == "Specialist").astype(float) +
    0.02 * (appt_type == "Follow-up").astype(float) +
    0.03 * (days_until >= 14).astype(float) +
    0.08 * (prev_no_shows >= 1).astype(float) +
    0.12 * (prev_no_shows >= 2).astype(float) +
    (-0.10) * (sms_sent == "Yes").astype(float)
)
noise = np.random.normal(0, 0.05, n)
prob_no_show = np.clip(0.05 + base_prob + noise, 0, 0.95)
no_show = (np.random.rand(n) < prob_no_show).astype(int)

df = pd.DataFrame({
    "Age": age,
    "Gender": gender,
    "Appointment_Type": appt_type,
    "Days_Until_Appointment": days_until,
    "Previous_No_Shows": prev_no_shows,
    "SMS_Reminder_Sent": sms_sent,
    "No_Show": no_show
})

X = df.drop(columns=["No_Show"])
y = df["No_Show"]

numeric_features = ["Age", "Days_Until_Appointment", "Previous_No_Shows"]
categorical_features = ["Gender", "Appointment_Type", "SMS_Reminder_Sent"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(drop="first", sparse=False), categorical_features)
])

rf_pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

log_pipeline = Pipeline([
    ("pre", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_pipeline.fit(X_train, y_train)
log_pipeline.fit(X_train, y_train)

rf_pred = rf_pipeline.predict(X_test)
rf_prob = rf_pipeline.predict_proba(X_test)[:, 1]
log_pred = log_pipeline.predict(X_test)
log_prob = log_pipeline.predict_proba(X_test)[:, 1]

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest ROC AUC:", roc_auc_score(y_test, rf_prob))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("Logistic Regression ROC AUC:", roc_auc_score(y_test, log_prob))
print("Logistic Regression Classification Report:\n", classification_report(y_test, log_pred))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, log_pred))

joblib.dump(rf_pipeline, "no_show_rf_model.joblib")
joblib.dump(log_pipeline, "no_show_logistic_model.joblib")

def predict_no_show(age, gender, appointment_type, days_until, previous_no_shows, sms_sent, model_type="rf"):
    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Appointment_Type": appointment_type,
        "Days_Until_Appointment": days_until,
        "Previous_No_Shows": previous_no_shows,
        "SMS_Reminder_Sent": sms_sent
    }])
    model = rf_pipeline if model_type == "rf" else log_pipeline
    prob = model.predict_proba(input_df)[0][1]
    pred = "Yes" if prob >= 0.5 else "No"
    return pred, prob

example = predict_no_show(35, "Female", "General Checkup", 7, 1, "Yes", model_type="rf")
print("Example Prediction:", example[0], f"({example[1]*100:.2f}%)")
