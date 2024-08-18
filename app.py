pip install seaborn
streamlit run app.py
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import time
import streamlit as st

# Set a random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)

# Load dataset from CSV file
df = pd.read_csv('Battery_RUL_II.csv')
st.write("### Data Preview")
st.dataframe(df.head())

# Calculate Correlation Matrix
correlation_matrix = df.corr()

# Visualize the Correlation Matrix
st.write("### Correlation Matrix Heatmap")
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
st.pyplot(plt)

# Split features (X) and target variable (y)
X = df.drop(columns=['RUL'])
y = df['RUL']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train, y_train)

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest model
random_forest_model = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
start_time = time.time()
grid_search.fit(X_train, y_train)
training_time = time.time() - start_time

# Best parameters found by GridSearchCV
best_params = grid_search.best_params_
st.write(f"### Best Parameters: {best_params}")

# Make predictions using the best model
best_model = grid_search.best_estimator_
start_time = time.time()
y_test_pred = best_model.predict(X_test)
prediction_time = time.time() - start_time

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)

# Display evaluation metrics and timing information
st.write("### Evaluation Metrics for Random Forest Regression")
st.write(f"MSE: {mse}")
st.write(f"MAE: {mae}")
st.write(f"RMSE: {rmse}")
st.write(f"R2: {r2}")
st.write(f"Training Time: {training_time} seconds")
st.write(f"Prediction Time: {prediction_time} seconds")

# Plot actual vs. predicted RUL
st.write("### Actual vs. Predicted RUL")
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('Actual vs. Predicted RUL (Random Forest)')
st.pyplot(plt)

# Function to get user input using Streamlit
def get_user_input():
    st.write("### Input Battery Parameters")
    Cycle_Index = st.number_input("Cycle_Index", min_value=0.0, step=0.1)
    Discharge_Time = st.number_input("Discharge Time", min_value=0.0, step=0.1)
    Decrement = st.number_input("Decrement 3.6-3.4V", min_value=0.0, step=0.1)
    Max_Voltage_Discharge = st.number_input("Max. Voltage Discharge (V)", min_value=0.0, step=0.1)
    Min_Voltage_Charge = st.number_input("Min. Voltage Charge (V)", min_value=0.0, step=0.1)
    Time = st.number_input("Time at 4.15V (s)", min_value=0.0, step=0.1)
    Time_constant_current = st.number_input("Time constant current (s)", min_value=0.0, step=0.1)
    Charging_time = st.number_input("Charging time (s)", min_value=0.0, step=0.1)
    return [Cycle_Index, Discharge_Time, Decrement, Max_Voltage_Discharge, Min_Voltage_Charge, Time, Time_constant_current, Charging_time]

# Function to predict RUL using the trained model
def predict_rul(user_features, regressor):
    feature_names = ['Cycle_Index', 'Discharge_Time', 'Decrement', 'Max_Voltage_Discharge', 'Min_Voltage_Charge', 'Time', 'Time_constant_current', 'Charging_time']
    user_features_df = pd.DataFrame([user_features], columns=feature_names)
    predicted_rul = regressor.predict(user_features_df)
    return predicted_rul[0]

# Function to map RUL to battery health categories
def map_to_battery_health(rul):
    if rul >= 741:
        return 'Excellent Battery Health'
    elif rul >= 371:
        return 'Average Battery Health'
    else:
        return 'Low or Poor Battery Health'

# Function to suggest optimization strategies based on battery health category
def suggest_optimization(category):
    if category == 'Excellent Battery Health':
        suggestions = [
            "Regular Monitoring: Continuously monitor the battery’s performance parameters.",
            "Optimal Charging Practices: Avoid overcharging and deep discharging. Use smart chargers.",
            "Temperature Management: Ensure the battery operates within the optimal temperature range.",
            "Balanced Usage: Ensure cells are balanced if the battery is part of a pack.",
            "Software Updates: Keep battery management software updated."
        ]
    elif category == 'Average Battery Health':
        suggestions = [
            "Conditioning Cycles: Perform controlled charging and discharging cycles.",
            "Reduced Load: Lower the load on the battery whenever possible.",
            "Partial Charging: Maintain the battery’s state of charge between 20% and 80%.",
            "Routine Maintenance: Regularly clean and check connections and terminals.",
            "Usage Adjustments: Adjust usage patterns to avoid high-drain scenarios."
        ]
    else:  # Low or Poor Battery Health
        suggestions = [
            "Capacity Testing: Regularly test the battery’s capacity.",
            "Load Reduction: Significantly reduce the load to prevent sudden failures.",
            "Refurbishment: Consider battery refurbishment options.",
            "Preemptive Replacement: Replace the battery to avoid unexpected failures.",
            "Recycling: Ensure proper recycling procedures for disposed batteries."
        ]
    return suggestions

# Get user input for features
user_features = get_user_input()

# Predict RUL based on user input
if st.button("Predict RUL"):
    predicted_rul = predict_rul(user_features, best_model)
    st.write(f"### Predicted Remaining Useful Life (RUL): {predicted_rul}")

    # Map the predicted RUL to a battery health category
    predicted_health_category = map_to_battery_health(predicted_rul)
    st.write(f"### Battery Health Category: {predicted_health_category}")

    # Suggest optimization strategies based on the battery health category
    optimization_suggestions = suggest_optimization(predicted_health_category)
    st.write("### Optimization Strategies:")
    for suggestion in optimization_suggestions:
        st.write(f"- {suggestion}")
