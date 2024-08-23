#%%writefile app.py
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('random_forest_model.pkl')

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

def main():
    st.title("Lithium Battery RUL Prediction")

    # Get user input for features
    st.header("Enter Battery Parameters:")
    Cycle_Index = st.number_input("Cycle Index", min_value=0.0)
    Discharge_Time = st.number_input("Discharge Time", min_value=0.0)
    Decrement = st.number_input("Decrement 3.6-3.4V", min_value=0.0)
    Max_Voltage_Discharge = st.number_input("Max Voltage Discharge (V)", min_value=0.0)
    Min_Voltage_Charge = st.number_input("Min Voltage Charge (V)", min_value=0.0)
    Time = st.number_input("Time at 4.15V (s)", min_value=0.0)
    Time_constant_current = st.number_input("Time constant current (s)", min_value=0.0)
    Charging_time = st.number_input("Charging time (s)", min_value=0.0)

    user_features = [
        Cycle_Index, Discharge_Time, Decrement,
        Max_Voltage_Discharge, Min_Voltage_Charge, Time,
        Time_constant_current, Charging_time
    ]

    if st.button("Predict RUL"):
        predicted_rul = predict_rul(user_features, model)

        # Map the predicted RUL to a battery health category
        predicted_health_category = map_to_battery_health(predicted_rul)

        # Suggest optimization strategies based on the battery health category
        optimization_suggestions = suggest_optimization(predicted_health_category)

        # Display the results
        st.subheader("Results:")
        st.write(f"The Predicted Remaining Useful Life (RUL) of your battery is: {predicted_rul:.2f} cycles")
        st.write(f"Battery Health Category: {predicted_health_category}")

        st.subheader("Optimization Strategies:")
        for suggestion in optimization_suggestions:
            st.write(f"- {suggestion}")

if __name__ == '__main__':
    main()
