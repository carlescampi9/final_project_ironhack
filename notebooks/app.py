import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load the pre-trained model and scalers
model_price = joblib.load('models/price_model.pkl')  # Load model from models/
normalizer = joblib.load('scalers/normalizer.pkl')  # Load MinMaxScaler from scalers/
ohe = joblib.load('scalers/ohe.pkl')  # Load OneHotEncoder from scalers/

# Configure the Viridis color palette
viridis = cm.get_cmap('viridis')
norm = mcolors.Normalize(vmin=0, vmax=1)

st.title("Airbnb Price Prediction")

st.sidebar.header("Enter the listing details")

# User inputs
city = st.sidebar.selectbox("City", ["Amsterdam", "Athens", "Barcelona", "Berlin", "Budapest", "Lisbon", "London", "Paris", "Rome", "Vienna"])
room_type = st.sidebar.selectbox("Room Type", ["Private room", "Entire home/apt", "Shared room"])
person_capacity = st.sidebar.selectbox("Person Capacity", [1,2, 3, 4, 6, 5])
cleanliness_rating = st.sidebar.slider("Cleanliness Rating", 0.0, 10.0, 5.0)
bedrooms = st.sidebar.selectbox("Number of Bedrooms", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
dist = st.sidebar.slider("Distance to City Center (km)", 0.0, 100.0, 10.0)
metro_dist = st.sidebar.slider("Distance to Metro (km)", 0.0, 100.0, 5.0)
attr_index = st.sidebar.slider("Attraction Index", 0.0, 3000.0, 1500.0)
host_is_superhost = st.sidebar.checkbox("Is Superhost?")
multi = st.sidebar.checkbox("Multiple Listing?")
biz = st.sidebar.checkbox("Business Accommodation?")
weekend = st.sidebar.checkbox("Is Weekend?")

# Prepare categorical variables
categorical_nominal = pd.DataFrame([[room_type, host_is_superhost, multi, biz, weekend, city]],
                                   columns=["room_type", "host_is_superhost", "multi", "biz", "weekend", "city"])
categorical_transformed = ohe.transform(categorical_nominal)

# Prepare numerical variables
numerical_columns = np.array([[cleanliness_rating, dist, metro_dist, attr_index]])
numerical_transformed = normalizer.transform(numerical_columns)

# Combine all inputs
X_input = np.hstack(([np.log1p(person_capacity), bedrooms], numerical_transformed, categorical_transformed))

# Prediction
if st.sidebar.button("Predict Price"):
    log_price_predicted = model_price.predict([X_input])[0]
    price_predicted = np.expm1(log_price_predicted)  # Convert log price back to actual price
    
    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Estimated Price:** {price_predicted:.2f} € per night")
    
    # Visualize price with Viridis color
    fig, ax = plt.subplots()
    color = viridis(norm(price_predicted / 500))  # Normalize with an arbitrary price reference
    ax.barh(["Estimated Price"], [price_predicted], color=color)
    ax.set_xlabel("Price (€)")
    ax.set_title("Predicted Price Visualization")
    st.pyplot(fig)

    