import streamlit as st
import pandas as pd

st.set_page_config(page_title="Drug Repurposing Demo", layout="wide")

st.title("Drug Repurposing Dashboard")

# Load CSV
data = pd.read_csv("drugs.csv")

# Show all data
st.subheader("All Drugs")
st.dataframe(data)

# Filter by target protein
target = st.selectbox("Select Target Protein", data["Target"].unique())
filtered = data[data["Target"] == target]

st.subheader(f"Drugs targeting {target}")
st.dataframe(filtered)

# Example: Top 3 most potent drugs (lowest IC50)
top_drugs = filtered.nsmallest(3, "IC50_nM")
st.subheader(f"Top 3 potent drugs for {target}")
st.table(top_drugs)
