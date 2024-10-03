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

def get_top_active_movers():
    try:
        url = "https://finance.yahoo.com/markets/stocks/most-active/"
        response = requests.get(url)
        response.raise_for_status()
        response.encoding = 'utf-8'  # Set the encoding to handle special characters
        soup = BeautifulSoup(response.text, 'html.parser')
        
        stock_list = []
        
        # Find the table containing the most active stocks
        table = soup.find('tbody', {'class': 'body yf-1dbt8wv'})
        if table:
            rows = table.find_all('tr')  # Get all rows
            for row in rows:
                cols = row.find_all('td')
                if cols:
                    symbol = cols[0].find('span', {'class': 'symbol'}).text.strip()
                    change_percent = cols[3].find('fin-streamer').text.strip()
                    volume = cols[4].find('fin-streamer').text.strip()
                    avg_volume = cols[5].text.strip()
                    stock_list.append({
                        'symbol': symbol,
                        'change_percent': change_percent,
                        'volume': volume,
                        'avg_volume': avg_volume
                    })
        return stock_list
    except Exception as e:
        print(f"Error fetching top active movers: {str(e)}")
        return []