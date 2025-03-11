import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load models and scalers
model_satisfaction = joblib.load('models/satis_model_reg.pkl')  # Modelo de Regresión
normalizer = joblib.load('scalers/normalizer.pkl')  # MinMaxScaler
ohe = joblib.load('scalers/ohe.pkl')  # OneHotEncoder

# Solución al error de Matplotlib (compatibilidad con versiones anteriores)
viridis = plt.get_cmap("viridis")

primary_color = mcolors.to_hex(viridis(0.6))  
background_color = mcolors.to_hex(viridis(0.2))  

title_html = f"""
    <h1 style='text-align: center; color: {primary_color};'>Guest Satisfaction Prediction</h1>
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

# Transformaciones
numerical_columns = pd.DataFrame([[cleanliness_rating, dist, metro_dist, attr_index]], 
                                 columns=["cleanliness_rating", "dist", "metro_dist", "attr_index"])

# Verificar que las columnas coincidan con las esperadas por el scaler
if list(numerical_columns.columns) != list(normalizer.feature_names_in_):
    st.error(f"Error: Features esperadas por el scaler: {list(normalizer.feature_names_in_)}")
    st.stop()

# Aplicar transformación con el scaler
numerical_transformed = normalizer.transform(numerical_columns)

# Transformar booleanos
host_is_superhost = st.sidebar.checkbox("Is Superhost?")
multi = st.sidebar.checkbox("Multiple Listing?")
biz = st.sidebar.checkbox("Business Accommodation?")
weekend = st.sidebar.checkbox("Is Weekend?")
host_is_superhost, multi, biz, weekend = map(int, [host_is_superhost, multi, biz, weekend])

# OneHotEncoding de variables categóricas
categorical_nominal = pd.DataFrame(
    [[room_type, host_is_superhost, multi, biz, weekend, city]],
    columns=["room_type", "host_is_superhost", "multi", "biz", "weekend", "city"]
)

try:
    categorical_transformed = ohe.transform(categorical_nominal)
except ValueError as e:
    st.error(f"Error en OneHotEncoder: {e}")
    st.stop()

categorical_transformed_df = pd.DataFrame(categorical_transformed, columns=ohe.get_feature_names_out())

# Variables manuales
numeric_manual = np.array([[person_capacity, bedrooms]])

# Combinar todas las variables para el modelo
X_input = np.hstack((
    numeric_manual,
    numerical_transformed,
    categorical_transformed_df.to_numpy()
))

# Convertir X_input a DataFrame con los nombres correctos para evitar errores en la predicción
feature_names = ["person_capacity", "bedrooms"] + list(normalizer.feature_names_in_) + list(ohe.get_feature_names_out())
X_input_df = pd.DataFrame(X_input, columns=feature_names)

# Botón de predicción
if st.sidebar.button("Predict Guest Satisfaction"):
    try:
        satisfaction_predicted = model_satisfaction.predict(X_input_df)[0]

        result_html = f"""
        <div style='text-align: center; padding: 20px; background-color: {background_color}; border-radius: 10px;'>
            <h2 style='color: white;'>Estimated Guest Satisfaction</h2>
            <h1 style='color: {primary_color}; font-size: 48px;'>{satisfaction_predicted:.2f}</h1>
            <p style='color: white; font-size: 18px;'>out of 100</p>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error en predicción: {e}")
