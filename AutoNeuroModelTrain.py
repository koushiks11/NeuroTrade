import json
import csv
import requests

# Fetch JSON data from the API
response = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=RELIANCE.BSE&outputsize=full&apikey=demo")
data = response.json()

# Extract time series data
time_series_data = data["Time Series (Daily)"]

# Sort the dates
sorted_dates = sorted(time_series_data.keys())

# Define CSV file name
csv_filename = "reliance_stock_data.csv"

# Define CSV header
csv_header = [
    "Date", "open", "high", "low", "close", "adjusted close", "volume", "dividend amount", "split coefficient"
]

# Write data to CSV file
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=csv_header)
    
    # Write header
    writer.writeheader()
    
    # Write data
    for date in sorted_dates:
        values = time_series_data[date]
        row = {
            "Date": date,
            "open": values["1. open"],
            "high": values["2. high"],
            "low": values["3. low"],
            "close": values["4. close"],
            "adjusted close": values["5. adjusted close"],
            "volume": values["6. volume"],
            "dividend amount": values["7. dividend amount"],
            "split coefficient": values["8. split coefficient"]
        }
        writer.writerow(row)

print(f"Data has been written to {csv_filename}")


import pandas as pd
import numpy as np
import tensorflow as tf

### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

df=pd.read_csv('reliance_stock_data.csv')
df = df[-1654:]
df1=df.reset_index()['high']

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))

##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, X=0,1,2,3-----99   Y=100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t,t+1,t+2..t+99 and Y=t+100

time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

print(X_train.shape), print(y_train.shape)
print(X_test.shape), print(ytest.shape)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

model.save("stock_prediction_model.h5")
np.savez("scaler_parameters.npz", min_=scaler.min_, scale_=scaler.scale_)


import datetime

# Function to read the last run time from file
def read_last_run_time(filename):
    try:
        with open(filename, 'r') as file:
            last_run_time = file.read()
            return last_run_time
    except FileNotFoundError:
        return None

# Function to append current time to file
def append_current_time(filename):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, 'a') as file:
        file.write(current_time + '\n')  # Append current time to a new line

# File to store last run times
filename = "last_run_times.txt"

# Read the last run time
last_run_time = read_last_run_time(filename)

if last_run_time:
    print("Last run times:\n" + last_run_time)
else:
    print("No previous run recorded.")

# Append current time to file
append_current_time(filename)


