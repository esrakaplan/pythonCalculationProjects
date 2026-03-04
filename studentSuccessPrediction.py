"""
Logistic Regression and Random Forest
Performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
Cross-validation and model comparison
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score
)

print("="*60)
print("PROJECT 3: Student Success Prediction (Classification)")
print("="*60)

# ===== STEP 1: Create Dataset =====
print("\n[STEP 1] Generate Training Dataset")
print("-" * 60)

np.random.seed(42)

n_students = 200

data = pd.DataFrame({
    'Class_Attendance': np.random.randint(40, 100, n_students),
    'Study_Hours_Per_Week': np.random.randint(1, 10, n_students),
    'Homework_Completion_Rate': np.random.randint(20, 100, n_students),
    'Exam_Average': np.random.randint(30, 100, n_students),
    'GPA': np.random.uniform(1.0, 4.0, n_students),
    'Library_Visits_Per_Month': np.random.randint(0, 30, n_students),
})

# Target variable (1 = Success, 0 = Fail)
data['Success'] = (
    (data['Exam_Average'] > 70) &
    (data['GPA'] > 2.5)
).astype(int)

print(f"Dataset Shape: {data.shape}")
print("\nTarget Distribution:")
print(data['Success'].value_counts())
print(f"Success Rate: %{(data['Success'].mean()*100):.1f}")

print("\nFirst 10 Rows:")
print(data.head(10))


# ===== STEP 2: Exploratory Data Analysis =====
print("\n[STEP 2] Exploratory Data Analysis (EDA)")
print("-" * 60)

print("\nDescriptive Statistics:")
print(data.describe().round(2))

print("\nCorrelation Matrix:")
correlation_matrix = data.corr().round(3)
print(correlation_matrix)

print("\nSuccessful Students (Mean Values):")
print(data[data['Success'] == 1][
    ['Class_Attendance', 'Study_Hours_Per_Week', 'GPA']
].mean().round(2))

print("\nUnsuccessful Students (Mean Values):")
print(data[data['Success'] == 0][
    ['Class_Attendance', 'Study_Hours_Per_Week', 'GPA']
].mean().round(2))


# ===== STEP 3: Preprocessing =====
print("\n[STEP 3] Data Preprocessing")
print("-" * 60)

X = data.drop('Success', axis=1)
y = data['Success']

print(f"Feature Matrix (X): {X.shape}")
print(f"Target Vector (y): {y.shape}")

print(f"\nMissing Values: {X.isnull().sum().sum()}")

#test_size=0.2 -> %80:training, %20 test. Learn from train value, evaluate with test value.
#random_state=42 shuffle data, reproducible results
# stratify=y -> if data : %30 success # %70 fail. Same rate preserved in train & test. Preserve the target rate.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print(f"\nTraining Set: {X_train.shape}")
print(f"Test Set: {X_test.shape}")


# ===== STEP 4: Feature Scaling =====
print("\n[STEP 4] Standardization")
print("-" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaling Completed")
print(f"Mean (Scaled Train Data): {X_train_scaled.mean(axis=0).round(3)}")
print(f"Std (Scaled Train Data): {X_train_scaled.std(axis=0).round(3)}")


# ===== STEP 5: Logistic Regression Model =====
print("\n[STEP 5] Model 1: Logistic Regression")
print("-" * 60)

log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)
y_prob_log = log_model.predict_proba(X_test_scaled)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_log):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_log):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_log):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_log):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log, target_names=['Fail', 'Success']))

print("\nModel Coefficients (Feature Importance):")
for i, col in enumerate(X.columns):
    print(f"{col}: {log_model.coef_[0][i]:.4f}")


# ===== STEP 6: Cross Validation =====
print("\n[STEP 6] Cross-Validation (5-Fold)")
print("-" * 60)

cv_scores = cross_val_score(log_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores.round(4)}")
print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


# ===== STEP 7: Random Forest Model =====
print("\n[STEP 7] Model 2: Random Forest")
print("-" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")

print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)


# ===== STEP 8: Model Comparison =====
print("\n[STEP 8] Model Comparison")
print("-" * 60)

comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
    'Logistic Regression': [
        accuracy_score(y_test, y_pred_log),
        precision_score(y_test, y_pred_log),
        recall_score(y_test, y_pred_log),
        f1_score(y_test, y_pred_log),
        roc_auc_score(y_test, y_prob_log)
    ],
    'Random Forest': [
        accuracy_score(y_test, y_pred_rf),
        precision_score(y_test, y_pred_rf),
        recall_score(y_test, y_pred_rf),
        f1_score(y_test, y_pred_rf),
        roc_auc_score(y_test, y_prob_rf)
    ]
}).round(4)

print(comparison)


# ===== STEP 9: Predictions on New Students =====
print("\n[STEP 9] Predict New Students")
print("-" * 60)

new_students = pd.DataFrame({
    'Class_Attendance': [85, 50, 95],
    'Study_Hours_Per_Week': [7, 2, 9],
    'Homework_Completion_Rate': [90, 40, 100],
    'Exam_Average': [80, 45, 95],
    'GPA': [3.5, 1.8, 3.9],
    'Library_Visits_Per_Month': [20, 2, 25]
})

predictions = rf_model.predict(new_students)
prediction_probs = rf_model.predict_proba(new_students)

print("\nNew Students Features:")
print(new_students)

print("\nPrediction Results (Random Forest):")
for i, pred in enumerate(predictions):
    status = "Success" if pred == 1 else "Fail"
    confidence = prediction_probs[i][pred]
    print(f"Student {i+1}: {status} (Confidence: %{confidence*100:.1f})")


# ===== OUTPUT =====
print("\n[OUTPUT] Save Results")
print("-" * 60)

data.to_csv('student_data.csv', index=False, encoding='utf-8')
print("✓ Student dataset saved to 'student_data.csv'")

comparison.to_csv('model_comparison.csv', index=False, encoding='utf-8')
print("✓ Model comparison saved to 'model_comparison.csv'")

print("\n" + "="*60)
print("PROJECT 3 COMPLETED")
print("="*60)