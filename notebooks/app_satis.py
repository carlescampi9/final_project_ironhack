import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Cargar modelo y transformadores
model_satisfaction = joblib.load('models/satisfaction_model.pkl')
normalizer = joblib.load('scalers/normalizer.pkl')
ohe = joblib.load('scalers/ohe.pkl')

# Configurar la paleta de colores Viridis
viridis = cm.get_cmap('viridis')
norm = mcolors.Normalize(vmin=0, vmax=1)
primary_color = mcolors.to_hex(viridis(0.6))
background_color = mcolors.to_hex(viridis(0.2))

# 📌 Obtener los valores originales de MinMaxScaler
satisfaction_min = normalizer.data_min_[-1]  # Última columna usada en MinMaxScaler
satisfaction_max = normalizer.data_max_[-1]  # Última columna usada en MinMaxScaler

# Interfaz de la aplicación
title_html = f"""
    <h1 style='text-align: center; color: {primary_color};'>Airbnb Client Satisfaction Prediction</h1>
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

# 📌 Variables **no usadas en el modelo**, pero requeridas por MinMaxScaler
guest_satisfaction_overall = 85.0  # Placeholder para evitar errores
rest_index = 500.0  # Placeholder para evitar errores

# 📌 **Transformaciones necesarias**
numerical_columns = np.array([[
    cleanliness_rating, guest_satisfaction_overall, dist, metro_dist, attr_index, rest_index  # ✅ Normalizamos TODAS las 6
]])
numerical_transformed = normalizer.transform(numerical_columns)

# 📌 **Transformar valores booleanos**
host_is_superhost = int(st.sidebar.checkbox("Is Superhost?"))
multi = int(st.sidebar.checkbox("Multiple Listing?"))
biz = int(st.sidebar.checkbox("Business Accommodation?"))
weekend = int(st.sidebar.checkbox("Is Weekend?"))

# 📌 **Transformar con OneHotEncoder**
categorical_nominal = pd.DataFrame(
    [[room_type, host_is_superhost, multi, biz, weekend, city]],
    columns=["room_type", "host_is_superhost", "multi", "biz", "weekend", "city"]
)

try:
    categorical_transformed = ohe.transform(categorical_nominal)
except ValueError as e:
    st.error(f"Error in OneHotEncoder: {e}")
    st.stop()

categorical_transformed_df = pd.DataFrame(categorical_transformed, columns=ohe.get_feature_names_out())

# 📌 **Asegurar la misma forma para `person_capacity` y `bedrooms`**
numeric_manual = np.array([[np.log1p(person_capacity), bedrooms]])

# 📌 **Combinar todas las features asegurando la misma dimensión**
X_input = np.hstack((
    numeric_manual,                          # (1,2)
    numerical_transformed[:, [0, 2, 3, 4]],  # ✅ Usamos solo las 4 correctas
    categorical_transformed_df.to_numpy()    # (1, X)
))

# 📌 **Predicción de Client Satisfaction**
if st.sidebar.button("Predict Client Satisfaction"):
    try:
        normalized_prediction = model_satisfaction.predict(X_input)[0]  # ✅ Predicción en escala 0-1
        
        # **Desnormalizar el resultado**
        satisfaction_predicted = normalized_prediction * (satisfaction_max - satisfaction_min) + satisfaction_min

        # **Mostrar resultado**
        result_html = f"""
        <div style='text-align: center; padding: 20px; background-color: {background_color}; border-radius: 10px;'>
            <h2 style='color: white;'>Estimated Client Satisfaction</h2>
            <h1 style='color: {primary_color}; font-size: 48px;'>{satisfaction_predicted:.2f} / 100</h1>
            <p style='color: white; font-size: 18px;'>Based on input details</p>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error in prediction: {e}")
