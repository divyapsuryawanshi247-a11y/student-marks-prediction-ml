import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Student Marks Predictor", page_icon="📚")

@st.cache_resource
def train_model():
    data = pd.read_csv("student_marks.csv")
    X = data[["study_hours", "attendance_percent", "assignment_percent", "previous_marks"]]
    y = data["final_marks"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return model, mse, r2

st.title("📚 Student Marks Prediction")
st.write("Enter student details to predict the expected final marks.")

model, mse, r2 = train_model()

study_hours = st.number_input("Study Hours per Day", min_value=1.0, max_value=12.0, value=5.0, step=0.5)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0)
assignment = st.number_input("Assignment Score (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
previous = st.number_input("Previous Exam Marks", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

if st.button("Predict Marks"):
    input_data = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance_percent": attendance,
        "assignment_percent": assignment,
        "previous_marks": previous
    }])
    prediction = float(model.predict(input_data)[0])
    prediction = max(0, min(100, prediction))
    st.success(f"Predicted Final Marks: {prediction:.2f}/100")

st.subheader("Model Evaluation")
col1, col2 = st.columns(2)
col1.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
col2.metric("R² Score", f"{r2:.3f}")

st.caption("Model: Linear Regression | Dataset: student_marks.csv")
