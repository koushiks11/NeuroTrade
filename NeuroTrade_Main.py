import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import altair as alt
import plotly.graph_objs as go
import plotly.express as px



# Page configuration
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈📉",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# Add custom CSS to reduce the padding of the title container
st.markdown(
    """
    <style>
    .st-emotion-cache-1wmy9hl {
        margin-top: -50px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Stock Closing Price Predictor")
st.write("**Note: The input date should not be more than 30 days from today. Saturdays and Sundays are not valid input dates.**")

selected_stock = "IBM"
input_date = datetime.today()


# Load the saved model
model = load_model("stock_prediction_model.h5")

# Global variables
lst_output = None
plot_lst_output = None

# Function to calculate difference in days while skipping Saturdays and Sundays
def difference_in_days(input_date):
    reference_date = datetime.today()
    input_date = datetime.strptime(input_date, "%Y-%m-%d")
    delta = timedelta(days=1)
    if input_date >= reference_date:
        difference_in_days = 0
        while input_date > reference_date:
            if input_date.weekday() < 5:  # Monday to Friday
                difference_in_days += 1
            input_date -= delta
        return difference_in_days
    else:
        return -(reference_date - input_date).days

# Function to predict closing price for a given date
def predict_closing_price(input_date, selected_stock):
    global lst_output
    days = difference_in_days(input_date)
    if days == 0:
        df = pd.read_csv(f'{selected_stock}_stock_data.csv')
        recent_closing_price = df.iloc[-1]['close']  # Assuming 'close' is the column name for closing price
        st.write(f"**The most recent closing price: {recent_closing_price}**")
        return
    if days >= 30:
        st.write("**Input date should not be more than 30 days from 2022-12-30**")
        return
    
    elif days > 0 and days < 30:
        df = pd.read_csv(f'{selected_stock}_stock_data.csv').tail(100)
        df1 = df.reset_index()
    
        df1 = df1['high']
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
        op = round(lst_output[days-1][0],3)
        st.write(f"**Predicted Closing Price for {input_date}: {op}**")
        return lst_output
    else:
        df = pd.read_csv(f'{selected_stock}_stock_data.csv')
        try:
            fetched_date_high = df[df['Date'] == input_date]['high'].values[0]
        except IndexError:
            fetched_date_high = -1
        if fetched_date_high != -1:
            st.write(f"**The high value for {input_date} is {fetched_date_high}**")
        elif fetched_date_high == -1:
            st.write(f"**The date corresponds to a stock market holiday or non-trading day. Choose other date**")


# Sidebar
with st.sidebar:
    st.title('📉📈 Stock Prediction Dashboard')
    
    selected_stock = st.selectbox("Select Stock", ["IBM", "RELIANCE"])
    

    # color_theme_list = ['blues', 'cividis', 'greens', 'inferno', 'magma', 'plasma', 'reds', 'rainbow', 'turbo', 'viridis']
    # selected_color_theme = st.selectbox('Select a color theme', color_theme_list)

    # Date input from user
    input_date = st.date_input("Enter a date", datetime.today())
    
    # Check if the input date is a Saturday or Sunday
    if input_date.weekday() >= 5:
        st.write("**Please select a weekday (Monday to Friday) as input.**")
    else:
        # Predict closing price when user submits the date
        if st.button("Predict Closing Price"):
            lst_output = predict_closing_price(str(input_date), selected_stock)
            # st.write(f"**Predicted Closing Price for {input_date}: {predicted_price}")


# Plotting the graph
if lst_output is not None:
    plot_lst_output = lst_output.copy()

# Separate section for plotting
df = pd.read_csv(f'{selected_stock}_stock_data.csv').tail(100)
df1 = df.reset_index()

df1 = df1['high']
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

x_input = df1[:,:1]
x_input = x_input.reshape(1,-1)
temp_input = list(x_input[0])
plot_lst_output = []
n_steps = 100

i = 0
days=30
while i < days:
    if len(temp_input) > 100:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        plot_lst_output.extend(yhat.tolist())
        i += 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        plot_lst_output.extend(yhat.tolist())
        i += 1
plot_lst_output = scaler.inverse_transform(plot_lst_output)

num_days_to_display = st.slider("Number of days to display", min_value=50, max_value=500, value=100, step=25)
df = pd.read_csv(f'{selected_stock}_stock_data.csv')
df1 = df.tail(num_days_to_display)  # Taking data from the last num_days_to_display days
df1 = df1.reset_index()['high']
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))
day_new = np.arange(1, num_days_to_display + 1)

# Creating arrays for actual and predicted data
actual_data = scaler.inverse_transform(df1[-num_days_to_display:]).flatten()
predicted_data = plot_lst_output.flatten()

# Creating dataframe for actual and predicted data
day_range_actual = np.arange(1, num_days_to_display + 1)
day_range_predicted = np.arange(num_days_to_display + 1, num_days_to_display + len(predicted_data) + 1)

actual_df = pd.DataFrame({'Day': day_range_actual, 'Closing Price': actual_data})
predicted_df = pd.DataFrame({'Day': day_range_predicted, 'Closing Price': predicted_data})

# Creating interactive line chart with Plotly Express
fig = px.line(title="Actual vs Predicted Closing Price")
fig.add_scatter(x=actual_df['Day'], y=actual_df['Closing Price'], mode='lines', name='Actual Data')
fig.add_scatter(x=predicted_df['Day'], y=predicted_df['Closing Price'], mode='lines', name='Predicted Data')
fig.update_layout(xaxis_title="Days", yaxis_title="Closing Price", legend_title="Data Type")

st.plotly_chart(fig)

st.write("""
**# Recent Data:
Taken From Quandl\n**
""")

# Load and display recent data
selected_df = pd.read_csv(f'{selected_stock}_stock_data.csv')
st.write(selected_df.tail(10))


