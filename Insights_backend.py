from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

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

def parse_news_data(news_tables):
    top_titles = []
    for ticker, news_table in news_tables.items():
        for row in news_table.findAll('tr'):
            title = row.a.text
            top_titles.append(title)
            if len(top_titles) == 10:
                break
        if len(top_titles) == 10:
            break
    
    print("Top 10 Titles:")
    for idx, title in enumerate(top_titles, 1):
        print(f"{idx}. {title}")

    return top_titles

def calculate_sentiment_scores(titles):
    vader = SentimentIntensityAnalyzer()
    compound_scores = []

    for title in titles:
        compound_score = vader.polarity_scores(title)['compound']
        compound_scores.append(compound_score)
        print(f"Title: {title} - Compound Score: {compound_score:.2f}")

    return compound_scores

def plot_sentiment(tickers, compound_scores):
    sentiments = ['Strongly Sell', 'Sell', 'Neutral', 'Buy', 'Strongly Buy']
    categories = pd.cut(compound_scores, bins=[-1, -0.5, -0.1, 0.1, 0.5, 1.0], labels=sentiments)

    df = pd.DataFrame({'Ticker': tickers * len(compound_scores), 'Compound Score': compound_scores, 'Sentiment': categories})
    df = df.iloc[:len(compound_scores)]  

    plt.figure(figsize=(12, 8))
    for ticker in tickers:
        subset = df[df['Ticker'] == ticker]
        sentiment_counts = subset['Sentiment'].value_counts()

        # Ensure all sentiments are present on the x-axis
        plt.bar(sentiments, [sentiment_counts.get(sentiment, 0) for sentiment in sentiments], color='white', alpha=0.5, label=f'{ticker} - Sentiment Category')

        # Highlight only the maximum sentiment category
        # max_sentiment = sentiment_counts.idxmax()
        # max_idx = sentiments.index(max_sentiment)
        # plt.bar(sentiments[max_idx], sentiment_counts[max_idx], color='green', alpha=0.5, label=f'{ticker} - Max Sentiment Category')

        # Plot sentiment scores
        plt.plot(subset['Sentiment'], subset['Compound Score'], marker='o', linestyle='-', label=f'{ticker} - Sentiment Score')

    plt.ylim(-1, 1)
    plt.ylabel('Score')
    plt.xlabel('Sentiment')
    plt.title('Sentiment Analysis with Highlighted Max Sentiment Category')
    plt.legend()
    plt.grid(True)

    # Calculate and display mean value
    mean_score = sum(compound_scores) / len(compound_scores)
    plt.axhline(y=mean_score, color='blue', linestyle='--', label=f'Mean Score: {mean_score:.2f}')
    plt.text(len(sentiments) - 1, mean_score, f'Mean Score: {mean_score:.2f}', ha='right', va='bottom')

    plt.show()

def main(value):
    tickers = [value]
    news_tables = get_news_data(tickers)
    titles = parse_news_data(news_tables)
    compound_scores = calculate_sentiment_scores(titles)
    plot_sentiment(tickers, compound_scores)

#Example
if __name__ == "__main__":
    main('AMD')
