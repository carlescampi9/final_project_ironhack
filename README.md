# Airbnb Price & Customer Satisfaction Prediction

## Overview
This project leverages machine learning to predict two key factors for Airbnb listings:
1. **Price** – Estimating the listing price based on various features such as geographic location, property characteristics, and amenities.
2. **Customer Satisfaction** – Predicting guest satisfaction scores using factors like host response rate, neighborhood quality, and previous reviews.

To achieve this, we built a robust pipeline that includes data preprocessing, exploratory data analysis (EDA), hypothesis testing, model selection, hyperparameter tuning, and deployment through a **Streamlit web application**.

---

## Project Structure

### 1. Data (`/data/`)
Contains all the datasets used in the project.
- **`raw/`** – Original uncleaned data files.
- **`clean/`** – Processed and merged datasets used for modeling.

### 2. Notebooks (`/notebooks/`)
Jupyter notebooks that document the data pipeline step by step:
- `Data Cleaning and Merging.ipynb` – Handles missing values, feature engineering, and data transformation.
- `EDA.ipynb` – Exploratory Data Analysis, including visualizations and statistical insights.
- `Hypothesis Testing.ipynb` – Investigates the statistical significance of various factors affecting price and satisfaction.
- `ML.ipynb` – Includes all machine learning models, hyperparameter tuning, and performance evaluation.

### 3. Models (`/models/`)
Stores trained machine learning models in `.pkl` format for deployment.
- Regression and **Adaptive Boosting models** were used for predictions.

### 4. Scalers (`/scalers/`)
Contains preprocessing tools stored as `.pkl` files:
- **MinMax Scaler** – Used to normalize numerical features.
- **OneHot Encoder** – Encodes categorical variables for machine learning models.

### 5. Figures (`/figures/`)
All visualizations from **EDA and ML modeling**, including:
- Correlation heatmaps
- Feature importance plots
- Model performance graphs

### 6. Web Applications (`/app_*.py`)
Deployed using **Streamlit**, allowing users to interact with the models:
- `app_reg.py` → Predicts Airbnb listing prices.
- `app_reg_satis.py` → Predicts Airbnb Client Satisfaction.

### 7. Slides (`/slides/`)
Presentation slides summarizing the project.
- **Includes a link to the deployed Streamlit apps** for real-time predictions.

---

## Methodology

### 1. Data Preprocessing
- Merged multiple datasets, removed outliers, and handled missing values.
- Engineered features such as location-based metrics, listing amenities, and host quality scores.
- Applied **MinMax Scaling** and **OneHot Encoding** for model compatibility.

### 2. Exploratory Data Analysis (EDA)
- Identified key drivers of price and customer satisfaction.
- Visualized trends across different cities, property types, and host behaviors.
- Used correlation matrices to understand feature relationships.

### 3. Hypothesis Testing
- Analyzed if location, property type, and review scores significantly impact price and satisfaction.
- Used **t-tests** and **ANOVA** to validate statistical assumptions.

### 4. Machine Learning Models

#### **Price Prediction Models**
- Linear Regression
- Bagging
- Random Forest Regressor
- Adaptive Boosting (**Best performance but overfitted**)
- Gradient Boosting

#### **Customer Satisfaction Models**
- Linear Regression
- Bagging
- Random Forest Regressor
- Adaptive Boosting (**Best performance but overfitted**)
- Gradient Boosting

- **Evaluated models based on RMSE, R², and classification metrics.**

### 5. Model Deployment
- Saved best models as `.pkl` files for deployment.
- Built **Streamlit web apps** to allow users to input features and get predictions.

---

## Results & Insights
- **Location and property type** were the strongest predictors of price.
- **Host response rate, cleanliness, and number of reviews** significantly influenced customer satisfaction.
- **Adaptive Boosting models outperformed** traditional regression/classification models.
- **The deployed Streamlit apps provide an easy-to-use interface** for real-time predictions.

---

## How to Use the Web App
1. Visit the **Streamlit app link** (provided in slides).
2. Input listing details (e.g., location, characteristics of the listing…).
3. Get an instant **price and customer satisfaction score prediction**.

---

## Future Improvements
- **Correcting overfitting** in some models and evaluating new R² scores for improved accuracy.
- **Adding real-time market data integration** (e.g., Airbnb API).

---

This project showcases how **machine learning can decode complex market dynamics** and provide actionable insights in the **Airbnb rental ecosystem**. 🚀
