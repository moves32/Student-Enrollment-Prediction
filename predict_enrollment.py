# Student Enrollment Prediction Model 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Create a dummy student_data.csv file for demonstration
dummy_data = {
    'GPA': [3.5, 2.8, 3.9, 3.2, 2.5, 3.0, 3.7, 2.9, 3.4, 2.7],
    'attendance_rate': [90, 75, 95, 88, 70, 85, 92, 78, 80, 65],
    'age': [18, 19, 18, 20, 19, 18, 21, 20, 19, 22],
    'gender': ['Male', 'Female', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'parent_income': [40000, 60000, 75000, 50000, 35000, 55000, 80000, 45000, 62000, 30000],
    'previous_enrollment': [1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    'will_enroll': [1, 0, 1, 1, 0, 1, 1, 0, 1, 0]
}
dummy_df = pd.DataFrame(dummy_data)
dummy_df.to_csv("student_data.csv", index=False)


# PREPROCESSING

# Convert categorical data to numeric
data['gender'] = data['gender'].map({'Male': 0, 'Female': 1})

# Select features (independent variables)
features = ['GPA', 'attendance_rate', 'age', 'gender',
            'parent_income', 'previous_enrollment']

X = data[features]

# Target variable (dependent variable)
y = data['will_enroll']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL TRAINING
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

#MODEL EVALUATION
predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# EXAMPLE PREDICTION FOR NEW STUDENT

new_student = pd.DataFrame([{
    "GPA": 3.2,
    "attendance_rate": 88,
    "age": 19,
    "gender": 1,  # Female
    "parent_income": 35000,
    "previous_enrollment": 1
}])

new_student_scaled = scaler.transform(new_student)

pred = model.predict(new_student_scaled)[0]
prob = model.predict_proba(new_student_scaled)[0][1]

print("\nPREDICTION FOR NEW STUDENT:")
print("Likely to enroll? =>", "YES" if pred == 1 else "NO")
print("Enrollment probability:", round(prob, 2))
