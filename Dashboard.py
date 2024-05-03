import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import altair as alt
import plotly.graph_objs as go
import plotly.express as px
import time


css = """
@keyframes fadeIn {
  from {
    opacity: 0;
  }
  to {
    opacity: 1;
  }
}

.overlay {
  position: fixed;
  width: 100%;
  height: 100%;
  top: 0;
  left: 0;
  background-color: rgba(0, 0, 0, 0.5);
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
}

.fade-in {
  animation: fadeIn 2s ease forwards;
}

.note {
    background-color: #f0f0f0;
    color: black;  /* Changed text color to black */
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 20px;
    max-width: 500px;
    position: relative;
}

.note::before {
    content: "";
    position: absolute;
    width: 20px;
    height: 20px;
    background-color: red;
    clip-path: polygon(100% 0%, 0% 50%, 100% 100%);
    top: 50%;
    left: -20px;
    transform: translateY(-50%);
}
"""

st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data()
def display_landing_page(dummy_input):
    # Landing page with fade-in animation
    with st.spinner("Loading..."):
        placeholder = st.empty()
        placeholder.markdown(
            """
            <style>
            @keyframes fadeIn {
            from {
                opacity: 0;
            }
            to {
                opacity: 1;
            }
            }

            .fade-in {
            animation: fadeIn 2s ease forwards;
            }
            </style>

            <div style='text-align: center; opacity: 0; animation: fadeIn 2s ease forwards;'>
                <h1 style='margin-top: 100px; font-size: 36px;'><b style='color: red'>NOTE</b>: This website is built For Educational Purposes Only</h1>
                <p style='font-size: 18px;'>This website and its content are intended for educational purposes only. The information provided here is for general informational purposes and should not be construed as investment advice. We do not endorse or recommend any specific stocks or trading strategies.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(3)
        placeholder.empty()


    placeholder = st.markdown(
        """
        <style>
        .overlay {
            position: fixed;
            width: 100%;
            height: 100%;
            top: 0;
            left: 0;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .note {
            background-color: #f0f0f0;
            color: black;  /* Changed text color to black */
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 20px;
            max-width: 500px;
            position: relative;
        }

        .note::before {
            content: "";
            position: absolute;
            width: 40px;
            height: 30px;
            background-color: red;
            clip-path: polygon(100% 0%, 0% 50%, 100% 100%);
            top: 50%;
            left: -40px;
            transform: translateY(-50%);
        }

        .fade-in {
        animation: fadeIn 2s ease forwards;
        }
        </style>

        <div class="overlay fade-in">
            <div class="note" style='text-align: center; opacity: 1;'>
                <p style='font-size: 18px;'>To use the dashboard, You have to select a stock from the dropdown menu in the sidebar. Choose a date and click 'Predict Closing Price' to see the predicted closing price. </p>
            </div>
        </div>
        """
        , unsafe_allow_html=True
    )
    time.sleep(5)
    placeholder.empty()


# Check if the landing page has already been displayed
if not display_landing_page("dummy_input"):
    # Display the landing page
    display_landing_page("dummy_input")
    

# Actual Dashboard code
alt.themes.enable("dark")

# Add custom CSS to apply padding-top: 25px
st.markdown(
    """
    <style>
    .block-container.st-emotion-cache-z5fcl4.ea3mdgi5 {
        padding-top: 20px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Stock Prediction Dashboard")

selected_stock = "IBM"
input_date = datetime.today()

# Load the saved model
model = load_model("stock_prediction_model.h5")

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
    days = difference_in_days(input_date)
    if days == 0:
        df = pd.read_csv(f'{selected_stock}_stock_data.csv')
        recent_closing_price = df.iloc[-1]['close']  # Assuming 'close' is the column name for closing price
        st.sidebar.write(f"**The most recent closing price: {recent_closing_price}**")
        return
    if days >= 30:
        st.sidebar.write("**Input date should not be more than 30 days from 2022-12-30**")
        return
    elif 0 < days < 30:
        df = pd.read_csv(f'{selected_stock}_stock_data.csv').tail(100)
        df1 = df['high']
        scaler = MinMaxScaler(feature_range=(0, 1))
        df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

        x_input = df1[:,:1]
        x_input = x_input.reshape(1, -1)
        temp_input = list(x_input[0])
        lst_output = []
        n_steps = 100

        i = 0
        while i < days:
            if len(temp_input) > 100:
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
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
        op = round(lst_output[days - 1][0], 3)
        st.sidebar.write(f"**Predicted Closing Price for {input_date}: {op}**")
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
            st.sidebar.write(f"**The date corresponds to a stock market holiday or non-trading day. Choose another date**")

# Sidebar
# st.sidebar.title('📉📈 ')
selected_stock = st.sidebar.selectbox("Select Stock", ["IBM", "Reliance", "Accenture", "Amazon", "Atlassian", "Dell", "GoldmanSachs", "Infosys", "JPMorgan", "Microsoft", "Nvidia", "Oracle", "Tesla", "Wipro"])
selected_stock = selected_stock.lower()
input_date = st.sidebar.date_input("Enter a date", datetime.today(),help="**Note: The input date should not be more than 30 days from today. Saturdays and Sundays are not valid input dates.**")

if input_date.weekday() >= 5:
    st.sidebar.write("**Please select a weekday (Monday to Friday) as input.**")
else:
    if st.sidebar.button("Predict Closing Price"):
        lst_output = predict_closing_price(str(input_date), selected_stock)




# Plotting the graph
col = st.columns((1.5, 4.5, 2), gap='medium')

df = pd.read_csv(f'{selected_stock}_stock_data.csv').tail(100)
df1 = df.reset_index()
df1 = df1['high']
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

x_input = df1[:,:1]
x_input = x_input.reshape(1, -1)
temp_input = list(x_input[0])
plot_lst_output = []
n_steps = 100

i = 0
days = 30
while i < days:
    if len(temp_input) > 100:
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
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
df = pd.read_csv(f'{selected_stock}_stock_data.csv').tail(num_days_to_display)
latest_date = pd.to_datetime(df['Date']).max()
df1 = df.reset_index()
scaler = MinMaxScaler(feature_range=(0, 1))
df1['Scaled High'] = scaler.fit_transform(np.array(df1['high']).reshape(-1, 1))

actual_data = df1['high'].values
predicted_data = plot_lst_output.flatten()

day_range_actual = pd.date_range(end=latest_date, periods=num_days_to_display).strftime('%Y-%m-%d')
day_range_predicted = pd.date_range(start=datetime.today() + timedelta(days=1), periods=len(predicted_data)).strftime('%Y-%m-%d')

actual_df = pd.DataFrame({'Date': day_range_actual, 'Closing Price': actual_data})
predicted_df = pd.DataFrame({'Date': day_range_predicted, 'Closing Price': predicted_data})

# Fetching opening prices
opening_data = df.set_index('Date')['open']
# Creating DataFrame for opening prices
opening_df = pd.DataFrame({'Date': day_range_actual, 'Opening Price': opening_data})

# Adding a toggle button for displaying only opening price plot
show_opening_price = st.checkbox("Show Opening Price Plot", False)

col = st.columns((1.5, 4.5, 2), gap='medium')

with col[0]:

    latest_closing_price = df1.iloc[-1]['high']
    latest_high_price = df1['high'].max()

    # Create containers for recent closing price and high price
    with st.container():
        st.markdown(
            f"<div style='border: 2px solid rgb(246, 51, 102); padding: 3px; text-align: center; color: white; width: auto; height: 150px; padding-top: 25%; border-radius: 10px; margin-top: 15px'>"
            f"<b>Recent Price</b>"
            f"<br>"
            f"<b><span style='font-size: 24px; color: rgb(255, 0, 0);'>{latest_closing_price}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    with st.container():
        st.markdown(
            f"<div style='border: 2px solid rgb(246, 51, 102); padding: 3px; text-align: center; color: white; width: auto; height: 150px; padding-top: 25%; border-radius: 10px; margin-top: 15px'>"
            f"<b>Highest Price</b>"
            f"<br>"
            f"<b><span style='font-size: 24px; color: Green;'>{latest_high_price}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

with col[1]:
    # Adding both opening and closing prices to the Plotly figure
    fig = px.line(title="Actual vs Predicted Closing")
    if show_opening_price:
        fig.add_scatter(x=opening_df['Date'], y=opening_df['Opening Price'], mode='lines', name='Actual Opening Price',text="Actual Opening Price", line=dict(color='orange'))
    fig.add_scatter(x=actual_df['Date'], y=actual_df['Closing Price'], mode='lines', name='Actual Closing Price',text="Actual Closing Price",line=dict(color='rgb(246, 51, 102)'))
    
    # Determine color based on predicted trend
    pred_trend_color = 'green' if predicted_data[-1] > actual_data[-1] else 'red'
    fig.add_scatter(x=predicted_df['Date'], y=predicted_df['Closing Price'], mode='lines', name='Predicted Closing Price',text="Predicted Closing Price", line=dict(color=pred_trend_color))
    
    fig.update_layout(xaxis_title="Date", yaxis_title="Price", legend_title="Data Type", title="Actual vs Predicted Closing and Opening Price")
    st.plotly_chart(fig)


st.write("""
**# Recent Data:
Taken From Quandl\n**
""")
st.write(df1.tail(10))

