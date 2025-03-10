import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load the pre-trained model and scalers
model_price = joblib.load('models/price_model.pkl')  # Load AdaBoostRegressor
normalizer = joblib.load('scalers/normalizer.pkl')  # Load MinMaxScaler
ohe = joblib.load('scalers/ohe.pkl')  # Load OneHotEncoder

# Configure the Viridis color palette
viridis = cm.get_cmap('viridis')
norm = mcolors.Normalize(vmin=0, vmax=1)

st.title("Airbnb Price Prediction")

st.sidebar.header("Enter the listing details")

# Sidebar inputs (valores reales, no normalizados ni dummificados)
city = st.sidebar.selectbox("City", ["Amsterdam", "Athens", "Barcelona", "Berlin", "Budapest", "Lisbon", "London", "Paris", "Rome", "Vienna"])
room_type = st.sidebar.selectbox("Room Type", ["Private room", "Entire home/apt", "Shared room"])
person_capacity = st.sidebar.selectbox("Person Capacity", [1, 2, 3, 4, 6, 5])
cleanliness_rating = st.sidebar.slider("Cleanliness Rating", 0.0, 10.0, 5.0)
bedrooms = st.sidebar.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
dist = st.sidebar.slider("Distance to City Center (km)", 0.0, 100.0, 10.0)
metro_dist = st.sidebar.slider("Distance to Metro (km)", 0.0, 100.0, 5.0)
attr_index = st.sidebar.slider("Attraction Index", 0.0, 3000.0, 1500.0)
host_is_superhost = st.sidebar.checkbox("Is Superhost?")
multi = st.sidebar.checkbox("Multiple Listing?")
biz = st.sidebar.checkbox("Business Accommodation?")
weekend = st.sidebar.checkbox("Is Weekend?")

### **📌 Transformaciones necesarias para que coincidan con el entrenamiento**
# Normalizar valores numéricos
numerical_columns = np.array([[cleanliness_rating, dist, metro_dist, attr_index]])
numerical_transformed = normalizer.transform(numerical_columns)

# Crear un DataFrame con los valores categóricos
categorical_nominal = pd.DataFrame(
    [[room_type, host_is_superhost, multi, biz, weekend, city]],
    columns=["room_type", "host_is_superhost", "multi", "biz", "weekend", "city"]
)

# Convertir valores booleanos a su versión correcta antes de transformarlos
categorical_nominal["host_is_superhost"] = categorical_nominal["host_is_superhost"].astype(str)
categorical_nominal["multi"] = categorical_nominal["multi"].astype(int)
categorical_nominal["biz"] = categorical_nominal["biz"].astype(int)
categorical_nominal["weekend"] = categorical_nominal["weekend"].astype(str)

# Aplicar OneHotEncoder asegurando que las columnas coinciden con el entrenamiento
categorical_transformed = ohe.transform(categorical_nominal)

# Convertir a DataFrame para garantizar que las columnas tienen el mismo orden
categorical_transformed_df = pd.DataFrame(categorical_transformed, columns=ohe.get_feature_names_out())

# Depuración: Ver las columnas generadas
st.write("Categorical transformed columns:", categorical_transformed_df.columns)

# Combinar todas las variables en el input del modelo
X_input = np.hstack((
    [np.log1p(person_capacity), bedrooms],  # Variables numéricas sin normalizar
    numerical_transformed,  # Variables normalizadas
    categorical_transformed_df.to_numpy()  # Variables categóricas correctamente transformadas
))

# **Predicción del precio**
if st.sidebar.button("Predict Price"):
    log_price_predicted = model_price.predict([X_input])[0]
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
