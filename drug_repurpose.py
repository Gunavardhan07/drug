import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import pickle

st.set_page_config(page_title="AI Drug Repurposing", layout="wide")
st.title("💊 AI-based Drug Repurposing Platform")

st.markdown("""
Upload a CSV file containing **SMILES strings** of drugs.  
The app will compute molecular features and predict drug repurposing potential.
""")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df)

    if 'SMILES' not in df.columns:
        st.error("CSV must contain a 'SMILES' column!")
    else:
        # Compute molecular descriptors
        def calc_features(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return pd.Series([np.nan]*5)
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hba = Descriptors.NumHAcceptors(mol)
            hbd = Descriptors.NumHDonors(mol)
            tpsa = Descriptors.TPSA(mol)
            return pd.Series([mw, logp, hba, hbd, tpsa])

        st.subheader("Calculating Molecular Descriptors...")
        df[['MolWt','LogP','HBA','HBD','TPSA']] = df['SMILES'].apply(calc_features)

        # Drop rows with NaN
        df.dropna(inplace=True)

        st.subheader("Feature Summary")
        st.dataframe(df[['MolWt','LogP','HBA','HBD','TPSA']].describe())

        # Scaling features
        features = ['MolWt','LogP','HBA','HBD','TPSA']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])

        # Dummy model: RandomForest (replace with real trained model)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        y_dummy = np.random.randint(0,2,len(df))
        clf.fit(X_scaled, y_dummy)
        df['Repurpose_Prediction'] = clf.predict(X_scaled)

        st.subheader("Predictions")
        st.dataframe(df[['Drug','Repurpose_Prediction']])

        # Visualize feature importance
        fig = px.bar(x=features, y=clf.feature_importances_, labels={'x':'Feature','y':'Importance'}, title="Feature Importance")
        st.plotly_chart(fig, use_container_width=True)

        # Interactive drug selection
        st.subheader("Check Individual Drug")
        drug_name = st.selectbox("Select Drug", df['Drug'].tolist())
        selected = df[df['Drug']==drug_name]
        st.write(selected)
