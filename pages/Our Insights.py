import streamlit as st
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import io
import matplotlib.pyplot as plt
import altair as alt
import plotly.graph_objs as go
import plotly.express as px
import base64
import urllib.request
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# nltk.download('vader_lexicon')

# Page configuration
st.set_page_config(
    page_title="Stock Sentiment Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
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

# Function to get news data from Finviz
def get_news_data(tickers):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}
    
    for ticker in tickers:
        url = finviz_url + ticker
        req = Request(url=url, headers={'user-agent': 'my-app'})

        try:
            response = urlopen(req)
            html = BeautifulSoup(response, features='html.parser')
            news_table = html.find(id='news-table')

            if news_table is not None:
                news_tables[ticker] = news_table
            else:
                print(f"No news table found for {ticker}.")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    return news_tables

# Function to parse news data
def parse_news_data(news_tables):
    top_titles = []
    for ticker, news_table in news_tables.items():
        for row in news_table.findAll('tr'):
            title_tag = row.a
            if title_tag:
                title = title_tag.text
                top_titles.append(title)
            if len(top_titles) == 10:
                break
        if len(top_titles) == 10:
            break
    
    return top_titles

# Function to calculate sentiment scores using Vader
def calculate_vader_sentiment_scores(titles):
    vader = SentimentIntensityAnalyzer()
    compound_scores = []

    for title in titles:
        compound_score = vader.polarity_scores(title)['compound']
        compound_scores.append(compound_score)

    return compound_scores

# Function to calculate sentiment scores using finBERT
def calculate_finbert_sentiment_scores(titles):
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone', do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
    model.eval()
    sentiment_scores = []

    for title in titles:
        inputs = tokenizer(title, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        sentiment_score = logits[0][predicted_class].item()
        sentiment_scores.append(sentiment_score)

    # Min-max normalization
    max_score = max(sentiment_scores)
    min_score = min(sentiment_scores)
    
    if max_score == min_score:
        normalized_scores = sentiment_scores  # If all scores are the same, no normalization needed
    else:
        normalized_scores = [(score - min_score) / (max_score - min_score) * 2 - 1 for score in sentiment_scores]

    return normalized_scores

# Function to plot sentiment using Vader
def plot_vader_sentiment(tickers, compound_scores):
    sentiments = ['Strongly Sell', 'Sell', 'Neutral', 'Buy', 'Strongly Buy']
    categories = pd.cut(compound_scores, bins=[-1, -0.5, -0.1, 0.1, 0.5, 1.0], labels=sentiments)

    df = pd.DataFrame({'Ticker': tickers * len(compound_scores), 'Compound Score': compound_scores, 'Sentiment': categories})
    df = df.iloc[:len(compound_scores)]  
    df = df.dropna(subset=['Sentiment'])
    plt.figure(figsize=(12, 8))
    for ticker in tickers:
        subset = df[df['Ticker'] == ticker]
        sentiment_counts = subset['Sentiment'].value_counts()

        # Ensure all sentiments are present on the x-axis
        plt.bar(sentiments, [sentiment_counts.get(sentiment, 0) for sentiment in sentiments], color='white', alpha=0.5, label=f'{ticker} - Sentiment Category')

        # Plot sentiment scores
        plt.plot(subset['Sentiment'], subset['Compound Score'], marker='o', linestyle='-', label=f'{ticker} - Sentiment Score')

    plt.ylim(-1, 1)
    plt.ylabel('Score')
    plt.xlabel('Sentiment')
    plt.title('Sentiment Analysis with Highlighted Max Sentiment Category (Vader)')
    plt.legend()
    plt.grid(True)

    # Calculate and display mean value
    mean_score = sum(compound_scores) / len(compound_scores)
    plt.axhline(y=mean_score, color='blue', linestyle='--', label=f'Mean Score: {mean_score:.2f}')
    plt.text(len(sentiments) - 1, mean_score, f'Mean Score: {mean_score:.2f}', ha='right', va='bottom')

    # Save the plot to an image buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()

    return img_buffer

# Function to plot sentiment using finBERT
def plot_finbert_sentiment(tickers, sentiment_scores):
    sentiments = ['Strongly Sell', 'Sell', 'Neutral', 'Buy', 'Strongly Buy']
    categories = pd.cut(sentiment_scores, bins=[-1, -0.5, -0.1, 0.1, 0.5, 1.0], labels=sentiments)

    df = pd.DataFrame({'Ticker': tickers * len(sentiment_scores), 'Sentiment Score': sentiment_scores, 'Sentiment': categories})
    df = df.iloc[:len(sentiment_scores)]  
    df = df.dropna(subset=['Sentiment'])
    plt.figure(figsize=(12, 8))
    for ticker in tickers:
        subset = df[df['Ticker'] == ticker]
        sentiment_counts = subset['Sentiment'].value_counts()

        plt.bar(sentiments, [sentiment_counts.get(sentiment, 0) for sentiment in sentiments], alpha=0.5, label=f'{ticker} - Sentiment Category')
        plt.plot(subset['Sentiment'].astype(str), subset['Sentiment Score'], marker='o', linestyle='-', label=f'{ticker} - Sentiment Score')

    plt.ylim(-1, 1)
    plt.ylabel('Score')
    plt.xlabel('Sentiment')
    plt.title('Sentiment Analysis with Highlighted Max Sentiment Category (finBERT)')
    plt.legend()
    plt.grid(True)

    mean_score = sum(sentiment_scores) / len(sentiment_scores)
    plt.axhline(y=mean_score, color='blue', linestyle='--', label=f'Mean Score: {mean_score:.2f}')
    plt.text(len(sentiments) - 1, mean_score, f'Mean Score: {mean_score:.2f}', ha='right', va='bottom')

    # Save the plot to an image buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()

    return img_buffer

# Streamlit app title
st.title('Stock Sentiment Analysis')

# Define ticker symbols and their corresponding values
ticker_mapping = {
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

# Get ticker symbols and their values from the dictionary
tickers_list = list(ticker_mapping.keys())
ticker_values = list(ticker_mapping.values())

# Dropdown menu for selecting ticker symbol
selected_ticker_index = st.selectbox('Select Stock', tickers_list)

# Toggle button for selecting sentiment analysis method
sentiment_method = st.radio('Select Sentiment Analysis Method', ('Vader', 'finBERT'))

# Checkbox to toggle display of compound scores
display_scores = st.checkbox('Display Compound Scores')

# Button to trigger sentiment analysis
analyze_button_clicked = st.button('Analyze Sentiment')
if analyze_button_clicked:
    with st.spinner('Analyzing sentiment...'):
        if sentiment_method == 'Vader':
            # Call the backend function with the selected ticker value using Vader
            img_buffer = plot_vader_sentiment([ticker_mapping[selected_ticker_index]], calculate_vader_sentiment_scores(parse_news_data(get_news_data([ticker_mapping[selected_ticker_index]]))))
            st.image(img_buffer, caption='Sentiment Analysis Plot (Vader)', use_column_width=True)

            # If the checkbox is checked, display titles with compound scores
            if display_scores:
                st.write('Titles with Compound Scores:')
                titles = parse_news_data(get_news_data([ticker_mapping[selected_ticker_index]]))
                compound_scores = calculate_vader_sentiment_scores(titles)

                # Create a DataFrame for displaying titles and compound scores
                df_titles = pd.DataFrame({'Serial No.': list(range(1, len(titles) + 1)), 'Title': titles, 'Compound Score': compound_scores})
                st.table(df_titles.set_index('Serial No.', drop=True))  # Display titles in a table format without the index column
            else:
                st.write('### News Articles')
                titles_only = parse_news_data(get_news_data([ticker_mapping[selected_ticker_index]]))
                # Create a DataFrame for displaying titles
                df_titles = pd.DataFrame({'Serial No.': list(range(1, len(titles_only) + 1)), 'Title': titles_only})
                st.table(df_titles.set_index('Serial No.', drop=True))  # Display titles in a table format without the index column

        elif sentiment_method == 'finBERT':
            # Call the backend function with the selected ticker value using finBERT
            img_buffer = plot_finbert_sentiment([ticker_mapping[selected_ticker_index]], calculate_finbert_sentiment_scores(parse_news_data(get_news_data([ticker_mapping[selected_ticker_index]]))))
            st.image(img_buffer, caption='Sentiment Analysis Plot (finBERT)', use_column_width=True)

            # Display titles with compound scores for finBERT
            st.write('Titles with Compound Scores (finBERT):')
            titles = parse_news_data(get_news_data([ticker_mapping[selected_ticker_index]]))
            compound_scores = calculate_finbert_sentiment_scores(titles)

            # Create a DataFrame for displaying titles and compound scores
            df_titles = pd.DataFrame({'Serial No.': list(range(1, len(titles) + 1)), 'Title': titles, 'Compound Score': compound_scores})
            st.table(df_titles.set_index('Serial No.', drop=True))  # Display titles in a table format without the index column