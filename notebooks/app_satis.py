import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load models and scalers
model_satisfaction = joblib.load('models/satisfaction_model.pkl')  # ✅ AdaBoostRegressor para satisfacción
normalizer = joblib.load('scalers/normalizer.pkl')  # ✅ MinMaxScaler
ohe = joblib.load('scalers/ohe.pkl')  # ✅ OneHotEncoder

# Configure the Viridis color palette
viridis = cm.get_cmap('viridis')
norm = mcolors.Normalize(vmin=0, vmax=1)
primary_color = mcolors.to_hex(viridis(0.6))  
background_color = mcolors.to_hex(viridis(0.2))  

title_html = f"""
    <h1 style='text-align: center; color: {primary_color};'>Airbnb Guest Satisfaction Prediction</h1>
"""
st.markdown(title_html, unsafe_allow_html=True)

st.sidebar.markdown(
    f"""
    <style>
        .sidebar .sidebar-content {{ background-color: {background_color}; color: white; }}
        .stButton>button {{ background-color: {primary_color}; color: white; font-size: 16px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Enter the listing details")

# Sidebar inputs
city = st.sidebar.selectbox("City", ohe.categories_[5])  
room_type = st.sidebar.selectbox("Room Type", ohe.categories_[0])
person_capacity = st.sidebar.selectbox("Person Capacity", [1, 2, 3, 4, 5, 6])
cleanliness_rating = st.sidebar.slider("Cleanliness Rating", 0.0, 10.0, 5.0)
bedrooms = st.sidebar.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
dist = st.sidebar.slider("Distance to City Center (km)", 0.0, 50.0, 10.0)
metro_dist = st.sidebar.slider("Distance to Metro (km)", 0.0, 50.0, 5.0)
attr_index = st.sidebar.slider("Attraction Index", 0.0, 2000.0, 1500.0)
host_is_superhost = st.sidebar.checkbox("Is Superhost?")
multi = st.sidebar.checkbox("Multiple Listing?")
biz = st.sidebar.checkbox("Business Accommodation?")
weekend = st.sidebar.checkbox("Is Weekend?")

# Normalizar los valores numéricos
numerical_columns = np.array([[cleanliness_rating, dist, metro_dist, attr_index]])
numerical_transformed = normalizer.transform(numerical_columns)

# Transform booleans
host_is_superhost, multi, biz, weekend = map(int, [host_is_superhost, multi, biz, weekend])

# Variables categóricas
categorical_nominal = pd.DataFrame(
    [[room_type, host_is_superhost, multi, biz, weekend, city]],
    columns=["room_type", "host_is_superhost", "multi", "biz", "weekend", "city"]
)

# OneHotEncoder
try:
    categorical_transformed = ohe.transform(categorical_nominal)
except ValueError as e:
    st.error(f"Error in OneHotEncoder: {e}")
    st.stop()

categorical_transformed_df = pd.DataFrame(categorical_transformed, columns=ohe.get_feature_names_out())

# Crear array de entrada correctamente estructurado
numeric_manual = np.array([[np.log1p(person_capacity), bedrooms]])  # ✅ Aplicar log a person_capacity

X_input = np.hstack((
    numeric_manual,
    numerical_transformed,
    categorical_transformed_df.to_numpy()
))

# **Depuración: Revisar X_input antes de predecir**
st.write("X_input shape:", X_input.shape)
st.write("X_input values:", X_input)

if st.sidebar.button("Predict Guest Satisfaction"):
    try:
        log_satisfaction_predicted = model_satisfaction.predict(X_input)[0]
        satisfaction_predicted = np.expm1(log_satisfaction_predicted)  # ✅ Convertir de log a escala normal

        result_html = f"""
        <div style='text-align: center; padding: 20px; background-color: {background_color}; border-radius: 10px;'>
            <h2 style='color: white;'>Estimated Guest Satisfaction</h2>
            <h1 style='color: {primary_color}; font-size: 48px;'>{satisfaction_predicted:.2f}</h1>
            <p style='color: white; font-size: 18px;'>out of 100</p>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error in prediction: {e}")
