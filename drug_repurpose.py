import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

# -------------------------
# 1. Load Drug Dataset
# -------------------------
drug_df = pd.read_csv('drug_data.csv')  # columns: drug_name, smiles

# -------------------------
# 2. Convert SMILES to Fingerprints
# -------------------------
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    else:
        return np.zeros(2048)

X = np.array([smiles_to_fp(s) for s in drug_df['smiles']])
y = np.random.randint(0,2,size=X.shape[0])  # For demo, use random labels

# -------------------------
# 3. Train Model (RandomForest)
# -------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# -------------------------
# 4. Streamlit UI
# -------------------------
st.title("AI-based Drug Repurposing")
st.write("Predict potential drugs for a new target protein")

user_smiles = st.text_input("Enter SMILES of candidate drug:")
if user_smiles:
    fp = smiles_to_fp(user_smiles).reshape(1, -1)
    score = model.predict_proba(fp)[0][1]
    st.write(f"Predicted repurposing probability: {score:.2f}")

st.write("Top 5 candidate drugs from dataset:")
if st.button("Show Top 5"):
    probs = model.predict_proba(X)[:,1]
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5 = drug_df.iloc[top5_idx]
    top5['score'] = probs[top5_idx]
    st.write(top5[['drug_name','score']])
