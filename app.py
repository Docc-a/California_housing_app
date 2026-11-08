import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="California House Price Predictor")
st.title("🏠 California House Price Predictor")
st.write("Enter the block group details below to predict the median house value.")

@st.cache_resource
def load_model():
    return joblib.load("california_knn_pipeline.joblib")

model = load_model()

col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("MedInc (median income, $10,000s)", 0.0, 20.0, 3.5)
    house_age = st.number_input("HouseAge (years)", 1.0, 60.0, 28.0)
    ave_rooms = st.number_input("AveRooms", 1.0, 20.0, 5.4)
    ave_bedrms = st.number_input("AveBedrms", 0.5, 10.0, 1.1)

with col2:
    population = st.number_input("Population", 1.0, 40000.0, 1400.0)
    ave_occup = st.number_input("AveOccup", 0.5, 20.0, 3.0)
    latitude = st.number_input("Latitude", 32.0, 42.0, 34.2)
    longitude = st.number_input("Longitude", -125.0, -114.0, -118.5)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude,
    }])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted median house value: ${prediction * 100000:,.0f}")
