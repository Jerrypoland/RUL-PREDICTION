"""
PREDICTION OF RUL OF LITHIUM BATTERY
USING RANDOM FOREST
"""
# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import time

# Set a random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
# Load dataset from CSV file
df=pd.read_csv('Battery_RUL II.csv')
df.head()
# Calculate Correlation Matrix
correlation_matrix = df.corr()

# Visualize the Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Split features (X) and target variable (y)
X = df.drop(columns=['RUL'])
y = df['RUL']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
random_forest_model = RandomForestRegressor()
random_forest_model.fit(X_train, y_train) # Train the model
 # Define the parameter grid
param_grid = {
    'n_estimators': [100, 300, 500],  # Number of trees
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}
# Initialize Random Forest model
random_forest_model = RandomForestRegressor(random_state=42)

# Initialize GridSearchCV with cross-validation
grid_search = GridSearchCV(estimator=random_forest_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
# Training time calculation
start_time = time.time()
grid_search.fit(X_train, y_train)
training_time = time.time() - start_time
# Best parameters found by GridSearchCV
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

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

# Print evaluation metrics and timing information
print("Evaluation Metrics for Random Forest Regression:")
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R2: {r2}")
print(f"Training Time: {training_time} seconds")
print(f"Prediction Time: {prediction_time} seconds")

# Plot actual vs. predicted RUL
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--')
plt.xlabel('Actual RUL')
plt.ylabel('Predicted RUL')
plt.title('Actual vs. Predicted RUL (Random Forest)')
plt.grid(True)
plt.show()

# Create pairplot for training data features vs. RUL
train_data_pairplot = pd.concat([X_train, y_train], axis=1)
sns.pairplot(train_data_pairplot)
plt.suptitle('Pairplot of Training Data Features vs. RUL', y=1.02)
plt.show()
# Create pairplot for test data features vs. RUL
test_data_pairplot = pd.concat([X_test, y_test], axis=1)
sns.pairplot(test_data_pairplot)
plt.suptitle('Pairplot of Test Data Features vs. RUL', y=1.02)
plt.show()
# Fit the model to the training data
random_forest_model.fit(X_train, y_train)
# Make predictions on training and test data
y_train_pred = random_forest_model.predict(X_train)
y_test_pred = random_forest_model.predict(X_test)

# Visualize training data with respect to RUL
plt.figure(figsize=(10, 6))

#y_train_pred = best_model.predict(X_train) # Predict on training data
# Scatter plot for training data
plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5, color='blue')
plt.xlabel('Actual RUL (Training)')
plt.ylabel('Predicted RUL (Training)')
plt.title('Actual vs. Predicted RUL (Training)')

plt.tight_layout()
plt.show()
# Visualize testing data with respect to RUL
plt.figure(figsize=(10, 6))
# Scatter plot for test data
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, color='red', alpha=0.5)
plt.xlabel('Actual RUL (Testing)')
plt.ylabel('Predicted RUL (Testing)')
plt.title('Actual vs. Predicted RUL (Testing)')

plt.tight_layout()
plt.show()
# Plot distribution of actual and predicted RUL
plt.figure(figsize=(8, 6))
plt.hist(y_test, bins=30, alpha=0.5, color='blue', label='Actual RUL')
random_forest_preds = best_model.predict(X_test)
plt.hist(random_forest_preds, bins=30, alpha=0.5, color='red', label='Predicted RUL')
plt.xlabel('RUL')
plt.ylabel('Frequency')
plt.title('Distribution of Actual and Predicted RUL (Random Forest)')
plt.legend()
plt.grid(True)
plt.show()
# Define function to plot learning curve
def plot_learning_curve(estimator, X, y):
    train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
                                                            scoring='neg_mean_squared_error', cv=5)
    train_scores_mean = -np.mean(train_scores, axis=1)
    val_scores_mean = -np.mean(val_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training error')
    plt.plot(train_sizes, val_scores_mean, 'o-', color='g', label='Validation error')
    plt.xlabel('Training examples')
    plt.ylabel('Negative Mean Squared Error')
    plt.title('Learning Curve (Random Forest)')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot learning curve for Random Forest
plot_learning_curve(random_forest_model, X, y)
# Function to get user input
def get_user_input():
    try:
        Cycle_Index = float(input("Enter Cycle_Index: "))
        Discharge_Time = float(input("Enter Discharge Time: "))
        Decrement = float(input("Enter Decrement 3.6-3.4V: "))
        Max_Voltage_Discharge = float(input("Enter Max. Voltage Discharge (V): "))
        Min_Voltage_Charge = float(input("Enter Min. Voltage Charge (V): "))
        Time = float(input("Enter Time at 4.15V (s): "))
        Time_constant_current = float(input("Enter Time constant current (s): "))
        Charging_time = float(input("Enter Charging time (s): "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return None
    return [Cycle_Index, Discharge_Time, Decrement, Max_Voltage_Discharge, Min_Voltage_Charge, Time, Time_constant_current, Charging_time]

# Function to predict RUL using the trained model
def predict_rul(user_features, regressor):
    # Convert user input into a DataFrame with the same feature names used during training
    feature_names = ['Cycle_Index', 'Discharge_Time', 'Decrement', 'Max_Voltage_Discharge', 'Min_Voltage_Charge', 'Time', 'Time_constant_current', 'Charging_time']
    user_features_df = pd.DataFrame([user_features], columns=feature_names)
    # Predict RUL using the trained model
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
if user_features is not None:
    # Predict RUL based on user input
    predicted_rul = predict_rul(user_features, best_model)

    # Map the predicted RUL to a battery health category
    predicted_health_category = map_to_battery_health(predicted_rul)

    # Suggest optimization strategies based on the battery health category
    optimization_suggestions = suggest_optimization(predicted_health_category)

    # Display the results
    print("\nThe Predicted Remaining Useful Life (RUL) of your battery is", predicted_rul)
    print("This estimate is a rough approximation and actual RUL can vary based on various factors including usage patterns, temperature, and charging habits.")
    print("The battery health category is:", predicted_health_category)
    print("Optimization Strategies:")
    for suggestion in optimization_suggestions:
        print(f"- {suggestion}")
