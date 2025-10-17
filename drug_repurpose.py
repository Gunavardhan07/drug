import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Drug Repurposing", layout="wide")
st.title("AI Drug Repurposing App")

uploaded_file = st.file_uploader("Upload your drug dataset CSV", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head(10))
    
    target_column = st.selectbox("Select the target column for prediction", data.columns)
    feature_columns = st.multiselect("Select feature columns", [col for col in data.columns if col != target_column])

    if st.button("Train Model"):
        if len(feature_columns) == 0:
            st.error("Please select at least one feature column")
        else:
            X = data[feature_columns]
            y = data[target_column]

            X = pd.get_dummies(X)
            if y.dtype == 'object':
                y = pd.factorize(y)[0]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"Model trained successfully. Accuracy: {acc:.2f}")

            uploaded_test_file = st.file_uploader("Upload new dataset to predict", type="csv", key="predict")
            if uploaded_test_file:
                test_data = pd.read_csv(uploaded_test_file)
                test_data_prepared = pd.get_dummies(test_data[feature_columns])
                test_data_prepared = test_data_prepared.reindex(columns=X.columns, fill_value=0)
                
                predictions = model.predict(test_data_prepared)
                test_data["Prediction"] = predictions
                st.dataframe(test_data)
                st.download_button("Download Predictions CSV", test_data.to_csv(index=False), file_name="predictions.csv")
else:
    st.info("Please upload your CSV dataset to proceed.")

