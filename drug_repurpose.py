import os
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------------
# Load CSV correctly
# ------------------------------
BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "drugs.csv")

# Load dataset
data = pd.read_csv(csv_path)

# ------------------------------
# Basic preprocessing
# ------------------------------
# Example: assuming CSV has 'feature1', 'feature2', ..., 'target'
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and compute accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Drug Repurposing Prediction")
st.write("Model Accuracy:", round(accuracy, 2))

# Input for user to predict new data
st.subheader("Predict for new input")
input_features = []
for col in X.columns:
    val = st.number_input(f"Enter value for {col}", value=float(data[col].mean()))
    input_features.append(val)

if st.button("Predict"):
    input_df = pd.DataFrame([input_features], columns=X.columns)
    prediction = model.predict(input_df)
    st.write("Predicted Drug Class / Outcome:", prediction[0])
