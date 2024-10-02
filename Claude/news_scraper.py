import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

def scrape_news(stock_symbol):
    url = f"https://www.google.com/search?q={stock_symbol}+stock+news"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract news headlines
    headlines = [h.get_text() for h in soup.find_all('h3')]
    return headlines

def analyze_sentiment(headlines):
    analyzer = SentimentIntensityAnalyzer()
    sentiments = [analyzer.polarity_scores(headline)['compound'] for headline in headlines]
    
    # Average sentiment score (positive if above 0.05, negative if below -0.05)
    avg_sentiment = np.mean(sentiments)
    return avg_sentiment

def get_news_sentiment(stock_symbol):
    headlines = scrape_news(stock_symbol)
    sentiment_score = analyze_sentiment(headlines)
    return sentiment_score