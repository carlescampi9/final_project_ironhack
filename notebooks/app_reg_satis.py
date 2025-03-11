import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load models and scalers
model_satisfaction = joblib.load('models/satis_model_reg.pkl')  # Regression Model
normalizer = joblib.load('scalers/normalizer.pkl')  # MinMaxScaler
ohe = joblib.load('scalers/ohe.pkl')  # OneHotEncoder

# Matplotlib compatibility adjustment
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

# Create DataFrame with numerical variables
numerical_columns = pd.DataFrame([[cleanliness_rating, dist, metro_dist, attr_index]], 
                                 columns=["cleanliness_rating", "dist", "metro_dist", "attr_index"])

# Add `guest_satisfaction_overall` and `rest_index` with dummy values
numerical_columns["guest_satisfaction_overall"] = 85  
numerical_columns["rest_index"] = 300  

# Reorder columns to match the scaler's expected input
numerical_columns = numerical_columns[["cleanliness_rating", "guest_satisfaction_overall", "dist", 
                                       "metro_dist", "attr_index", "rest_index"]]

# Apply MinMaxScaler transformation
numerical_transformed = normalizer.transform(numerical_columns)

# Transform boolean variables
host_is_superhost = st.sidebar.checkbox("Is Superhost?")
multi = st.sidebar.checkbox("Multiple Listing?")
biz = st.sidebar.checkbox("Business Accommodation?")
weekend = st.sidebar.checkbox("Is Weekend?")
host_is_superhost, multi, biz, weekend = map(int, [host_is_superhost, multi, biz, weekend])

# OneHotEncoding for categorical variables
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

# Manual numerical variables
numeric_manual = np.array([[person_capacity, bedrooms]])

# Combine all features for model input
X_input = np.hstack((
    numeric_manual,
    numerical_transformed,
    categorical_transformed_df.to_numpy()
))

# Get expected feature names from the trained model
expected_features = model_satisfaction.feature_names_in_

# Convert X_input to DataFrame with correct column names to avoid prediction errors
feature_names = ["person_capacity", "bedrooms"] + list(normalizer.feature_names_in_) + list(ohe.get_feature_names_out())
X_input_df = pd.DataFrame(X_input, columns=feature_names)

# Select only the columns expected by the model
X_input_df = X_input_df[expected_features]

# Prediction button
if st.sidebar.button("Predict Guest Satisfaction"):
    try:
        satisfaction_predicted = model_satisfaction.predict(X_input_df)[0]

        # Scale the value to percentage
        satisfaction_predicted = satisfaction_predicted * 100  

        # Ensure the maximum value is 100
        satisfaction_predicted = min(satisfaction_predicted, 100)

        result_html = f"""
        <div style='text-align: center; padding: 20px; background-color: {background_color}; border-radius: 10px;'>
            <h2 style='color: white;'>Estimated Guest Satisfaction</h2>
            <h1 style='color: {primary_color}; font-size: 48px;'>{satisfaction_predicted:.4f}</h1>
            <p style='color: white; font-size: 18px;'>out of 100</p>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error in prediction: {e}")
