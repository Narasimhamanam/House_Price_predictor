import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("house_price_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="🏡 House Price Predictor", layout="centered")

st.title("🏠 House Price Predictor")
st.markdown("Predict housing prices based on key features (using Ames dataset).")

LotArea = st.slider("📏 Lot Area (sq ft)", 1300, 20000, 8000)
YearBuilt = st.slider("🏗️ Year Built", 1872, 2010, 1980)
TotalBsmtSF = st.slider("🏚️ Basement Area (sq ft)", 0, 6000, 1000)
GrLivArea = st.slider("🏡 Above Ground Living Area (sq ft)", 334, 5642, 1500)
GarageCars = st.slider("🚗 Garage Capacity", 0, 4, 2)
OverallQual = st.selectbox("🏅 Overall Quality", list(range(1, 11)), index=5)

input_data = np.array([[LotArea, YearBuilt, TotalBsmtSF, GrLivArea, GarageCars, OverallQual]])
input_scaled = scaler.transform(input_data)
predicted_price = model.predict(input_scaled)[0]

if st.button("🔍 Predict Price"):
    st.success(f"💰 Estimated Price: ${predicted_price:,.2f}")
