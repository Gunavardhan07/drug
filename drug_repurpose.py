import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Drug Repurposing", layout="wide")

st.title("💊 AI-Based Drug Repurposing App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("drugs.csv")
    return df

data = load_data()
st.subheader("Drug Dataset")
st.dataframe(data)

# Train ML model
X = data[["Feature1", "Feature2", "Feature3"]]
y = data["Approved"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

st.subheader("Predict Drug Approval")

# User input
feature1 = st.number_input("Feature1", min_value=0.0, max_value=1.0, value=0.5)
feature2 = st.number_input("Feature2", min_value=0.0, max_value=1.0, value=0.5)
feature3 = st.number_input("Feature3", min_value=0.0, max_value=1.0, value=0.5)

if st.button("Predict"):
    prediction = model.predict([[feature1, feature2, feature3]])[0]
    if prediction == 1:
        st.success("✅ This drug is likely approved / repurposable!")
    else:
        st.error("❌ This drug is unlikely to be approved.")
