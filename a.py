import streamlit as st
import numpy as np
import pickle
import json

# Load model and supporting files
model = pickle.load(open('random_forest_model.pkl', 'rb'))
scaler = pickle.load(open('StandardScaler.pkl', 'rb'))
crops = json.load(open('crops.json', 'rb'))

# App Title & Styling
st.set_page_config(page_title="Crop Recommendation System", layout="centered")
st.markdown("""
    <style>
        .main-container {
            background-color: #f4f4f9;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            max-width: 800px;
            margin: auto;
        }
        .stTextInput > div > div > input {
            padding: 12px;
            font-size: 16px;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 1.2rem;
            padding: 12px 20px;
            border-radius: 8px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .form-spacing {
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌾 Crop Recommendation System")

# Add space between container and form
st.markdown("<div class='form-spacing'></div>", unsafe_allow_html=True)

# Form layout with 4 rows and 2 columns
with st.form("crop_form"):
    col1, col2 = st.columns(2)
    with col1:
        nitrogen = st.text_input("Nitrogen Ratio:")
    with col2:
        phosphorus = st.text_input("Phosphorus Ratio:")

    col3, col4 = st.columns(2)
    with col3:
        potassium = st.text_input("Potassium Ratio:")
    with col4:
        temperature = st.text_input("Temperature (°C):")

    col5, col6 = st.columns(2)
    with col5:
        humidity = st.text_input("Humidity (%):")
    with col6:
        ph_value = st.text_input("pH Value:")

    # Centered input for rainfall
    col7, col8, col9 = st.columns([1, 2, 1])
    with col8:
        rainfall = st.text_input("Rainfall (mm):")

    submit = st.form_submit_button("Get Crop Recommendation")

# Prediction and Output
if submit:
    try:
        features = np.array([[float(nitrogen), float(phosphorus), float(potassium),
                              float(temperature), float(humidity), float(ph_value), float(rainfall)]])
        features = scaler.transform(features)
        prediction = model.predict(features)[0]
        crop_name = crops[str(int(prediction))]

        st.success(f"✅ Recommended crop is: **{crop_name}**")

    except ValueError:
        st.error("🚫 Please enter valid numerical values for all fields.")
