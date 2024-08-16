#!pip install streamlit pyngrok
%%writefile app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Title of the app
st.title("Prediction of RUL of Lithium Battery Using Random Forest")

# Sidebar inputs
st.sidebar.header('User Input Features')

def get_user_input():
    Cycle_Index = st.sidebar.number_input('Cycle Index', min_value=0)
    Discharge_Time = st.sidebar.number_input('Discharge Time', min_value=0.0)
    Decrement = st.sidebar.number_input('Decrement 3.6-3.4V', min_value=0.0)
    Max_Voltage_Discharge = st.sidebar.number_input('Max Voltage Discharge', min_value=0.0)
    Min_Voltage_Charge = st.sidebar.number_input('Min Voltage Charge', min_value=0.0)
    Time = st.sidebar.number_input('Time at 4.15V', min_value=0.0)
    Time_constant_current = st.sidebar.number_input('Time constant current', min_value=0.0)
    Charging_time = st.sidebar.number_input('Charging time', min_value=0.0)
    
    user_data = {
        'Cycle_Index': Cycle_Index,
        'Discharge_Time': Discharge_Time,
        'Decrement': Decrement,
        'Max_Voltage_Discharge': Max_Voltage_Discharge,
        'Min_Voltage_Charge': Min_Voltage_Charge,
        'Time': Time,
        'Time_constant_current': Time_constant_current,
        'Charging_time': Charging_time
    }
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

# Load the dataset and train a Random Forest model (using sample data)
# Load your actual data and model here
model = RandomForestRegressor()
# Assuming you have pre-trained the model or load the model from a file
# For this example, let's just mock some data
X = pd.DataFrame(np.random.randn(100, 8), columns=user_input.columns)
y = np.random.randn(100)

model.fit(X, y)
prediction = model.predict(user_input)

st.subheader('Prediction')
st.write(f'The Predicted Remaining Useful Life (RUL) of your battery is: {prediction[0]:.2f}')
