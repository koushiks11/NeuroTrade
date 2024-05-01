import json
import csv
import requests

# List of stock symbols and corresponding names
stocks = {
    "IBM": "IBM",
    "Reliance": "RELI",
    "Accenture": "ACN",
    "Amazon": "AMZN",
    "Atlassian": "TEAM",
    "Dell": "DELL",
    "GoldmanSachs": "GS",
    "Infosys": "INFY",
    "JPMorgan": "JPM",
    "Microsoft": "MSFT",
    "Nvidia": "NVDA",
    "Oracle": "ORCL",
    "Tesla": "TSLA",
    "Wipro": "WIT"
}

# Define CSV header
csv_header = [
    "Date", "open", "high", "low", "close", "adjusted close", "volume", "dividend amount", "split coefficient"
]

# Fetch data for each stock symbol
for name, symbol in stocks.items():
    # Fetch JSON data from the API
    response = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey=0LWURRJDVQXKTZLK")
    data = response.json()
    
    # Extract time series data
    time_series_data = data["Time Series (Daily)"]
    
    # Sort the dates
    sorted_dates = sorted(time_series_data.keys())
    
    # Define CSV file name (in lowercase)
    csv_filename = f"{name.lower()}_stock_data.csv"
    
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
                "close": values["4. close"]
            }
            writer.writerow(row)
    
    print(f"Data for {name} has been written to {csv_filename}")
