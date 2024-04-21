import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = load_model("stock_prediction_model.h5")

# Function to scale the input date
def difference_in_days(input_date):
    reference_date = datetime.strptime("2022-12-30", "%Y-%m-%d")
    input_date = datetime.strptime(input_date, "%Y-%m-%d")
    difference_in_days = (input_date - reference_date).days
    return difference_in_days

# Function to predict closing price for a given date
def predict_closing_price(input_date):
    days = difference_in_days(input_date)
    if days > 30:
        return "Error: Input date is more than 30 days from the last available date"
    
    df = pd.read_csv('output.csv').tail(100)
    df1 = df.reset_index()['high']
    scaler = MinMaxScaler(feature_range=(0,1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

    x_input = df1[:,:1]
    x_input = x_input.reshape(1,-1)
    temp_input = list(x_input[0])
    lst_output = []
    n_steps = 100
    
    i = 0
    while i < days:
        if len(temp_input) > 100:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i += 1
    lst_output = scaler.inverse_transform(lst_output)
    return lst_output[days-1][0]

# Streamlit app
st.title("Stock Closing Price Predictor")

# Date input from user
input_date = st.date_input("Enter a date", datetime.today())

# Predict closing price when user submits the date
if st.button("Predict Closing Price"):
    predicted_price = predict_closing_price(str(input_date))
    st.write(f"Predicted Closing Price for {input_date}: {predicted_price}")
