import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load models and scalers
model_price = joblib.load('models/price_model.pkl')  #  AdaBoostRegressor
normalizer = joblib.load('scalers/normalizer.pkl')  #  MinMaxScaler
ohe = joblib.load('scalers/ohe.pkl')  #  OneHotEncoder

# Configure the Viridis color palette
viridis = cm.get_cmap('viridis')
norm = mcolors.Normalize(vmin=0, vmax=1)

st.title("Airbnb Price Prediction")

st.sidebar.header("Enter the listing details")

# Sidebar inputs (valores reales, no normalizados ni dummificados)
city = st.sidebar.selectbox("City", ohe.categories_[5])  #  Se toma de OneHotEncoder
room_type = st.sidebar.selectbox("Room Type", ohe.categories_[0])
person_capacity = st.sidebar.selectbox("Person Capacity", [1, 2, 3, 4, 6, 5])
cleanliness_rating = st.sidebar.slider("Cleanliness Rating", 0.0, 10.0, 5.0)
bedrooms = st.sidebar.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
dist = st.sidebar.slider("Distance to City Center (km)", 0.0, 100.0, 10.0)
metro_dist = st.sidebar.slider("Distance to Metro (km)", 0.0, 100.0, 5.0)
attr_index = st.sidebar.slider("Attraction Index", 0.0, 3000.0, 1500.0)
guest_satisfaction_overall = st.sidebar.slider("Guest Satisfaction (not used in model)", 0.0, 100.0, 85.0)  #  Solo para normalizar
rest_index = st.sidebar.slider("Restaurant Index (not used in model)", 0.0, 1000.0, 500.0)  #  Solo para normalizar
host_is_superhost = st.sidebar.checkbox("Is Superhost?")
multi = st.sidebar.checkbox("Multiple Listing?")
biz = st.sidebar.checkbox("Business Accommodation?")
weekend = st.sidebar.checkbox("Is Weekend?")

### ** Transformaciones necesarias para que coincidan con el entrenamiento**
# Normalizar valores numéricos (pasamos las 6 variables, aunque solo usaremos 4 en el modelo)
numerical_columns = np.array([[
    cleanliness_rating, 
    guest_satisfaction_overall,  #  Incluido aunque no se use en el modelo
    dist, 
    metro_dist, 
    attr_index, 
    rest_index  #  Incluido aunque no se use en el modelo
]])
numerical_transformed = normalizer.transform(numerical_columns)

#  **Transformar correctamente los valores booleanos**
host_is_superhost = int(host_is_superhost)  #  Convertir a 0 o 1
multi = int(multi)  #  Convertir a 0 o 1
biz = int(biz)  #  Convertir a 0 o 1
weekend = int(weekend)  #  Convertir a 0 o 1

# Crear un DataFrame con los valores categóricos
categorical_nominal = pd.DataFrame(
    [[room_type, host_is_superhost, multi, biz, weekend, city]],
    columns=["room_type", "host_is_superhost", "multi", "biz", "weekend", "city"]
)

#  **Transformar con OneHotEncoder**
try:
    categorical_transformed = ohe.transform(categorical_nominal)
except ValueError as e:
    st.error(f"Error in OneHotEncoder: {e}")
    st.stop()

# Convertir a DataFrame para garantizar que las columnas tienen el mismo orden
categorical_transformed_df = pd.DataFrame(categorical_transformed, columns=ohe.get_feature_names_out())

# Depuración: Ver las columnas generadas
st.write("Categorical transformed columns:", categorical_transformed_df.columns)

#  Asegurar que `person_capacity` y `bedrooms` tienen la misma forma
numeric_manual = np.array([[np.log1p(person_capacity), bedrooms]])

#  **Combinar todas las features asegurando la misma dimensión**
X_input = np.hstack((
    numeric_manual,                          #  (1,2)
    numerical_transformed[:, [0, 2, 3, 4]],  # Usamos solo las 4 variables que necesita el modelo
    categorical_transformed_df.to_numpy()    #  (1, X)
))

# **Predicción del precio**
if st.sidebar.button("Predict Price"):
    try:
        log_price_predicted = model_price.predict(X_input)[0]
        price_predicted = np.expm1(log_price_predicted)  # Convertir el logaritmo del precio al precio original

        # **Mostrar resultado**
        st.subheader("Prediction Results")
        st.write(f"**Estimated Price:** {price_predicted:.2f} € per night")

        # **Visualizar precio con la paleta Viridis**
        fig, ax = plt.subplots()
        color = viridis(norm(price_predicted / 500))  # Normalize with an arbitrary price reference
        ax.barh(["Estimated Price"], [price_predicted], color=color)
        ax.set_xlabel("Price (€)")
        ax.set_title("Predicted Price Visualization")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in prediction: {e}")
