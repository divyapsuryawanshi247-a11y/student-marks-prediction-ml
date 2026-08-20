import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("student_marks.csv")

X = data[["study_hours", "attendance_percent", "assignment_percent", "previous_marks"]]
y = data["final_marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Student Marks Prediction - Linear Regression")
print("--------------------------------------------")
print(f"MSE: {mean_squared_error(y_test, predictions):.2f}")
print(f"R2 Score: {r2_score(y_test, predictions):.3f}")

sample = pd.DataFrame([{
    "study_hours": 6,
    "attendance_percent": 85,
    "assignment_percent": 80,
    "previous_marks": 75
}])
print(f"Sample predicted marks: {model.predict(sample)[0]:.2f}")
