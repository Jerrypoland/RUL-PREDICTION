import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split

# Load the pre-trained models (assuming you have saved them previously)
# For demonstration, we'll retrain them in this example

# Load dataset from CSV file
df = pd.read_csv('Battery_RUL II.csv')

# Feature Engineering: Create polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(df.drop(columns=['RUL']))

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Principal Component Analysis (PCA) for feature reduction
pca = PCA(n_components=0.95)  # Keep 95% of the variance
X_pca = pca.fit_transform(X_scaled)

# Split features (X) and target variable (y)
X = pd.DataFrame(X_pca)
y = df['RUL']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Regressor
random_forest_model = RandomForestRegressor(random_state=42)
random_forest_model.fit(X_train, y_train)

# Initialize and train the Gradient Boosting Regressor
gradient_boosting_model = RandomForestRegressor(random_state=42)
gradient_boosting_model.fit(X_train, y_train)

# Combine the models using Voting Regressor
voting_regressor = VotingRegressor(estimators=[
    ('rf', random_forest_model),
    ('gb', gradient_boosting_model)
])

# Train the Voting Regressor
voting_regressor.fit(X_train, y_train)

# Streamlit application

st.title('Lithium Battery Remaining Useful Life (RUL) Prediction')

# Collect user input for each feature
st.header('Input features')

# Assuming your original dataset had features like 'feature1', 'feature2', 'feature3', etc.
input_data = {}
for feature_name in df.drop(columns=['RUL']).columns:
    input_data[feature_name] = st.number_input(f'Input {feature_name}', value=0.0)

# Convert input data to DataFrame
input_df = pd.DataFrame([input_data])

# Apply the same feature engineering steps
input_poly = poly.transform(input_df)
input_scaled = scaler.transform(input_poly)
input_pca = pca.transform(input_scaled)

# Predict RUL using the trained Voting Regressor
predicted_rul = voting_regressor.predict(input_pca)

# Display the prediction
st.subheader('Predicted Remaining Useful Life (RUL)')
st.write(f'{predicted_rul[0]:.2f} cycles')

# Run the app using `streamlit run app.py`
