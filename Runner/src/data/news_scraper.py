"""Fetches trending tickers from multiple sources
(Yahoo Finance, Stocktwits, ApeWisdom)
and prints the top 25 tickers for analysis by another code file.
"""

import asyncio
import json
import logging
import requests
import pandas as pd
from typing import Set, Optional, List, Dict
from datetime import datetime

import aiohttp
from bs4 import BeautifulSoup

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsScraper:
    """Scrapes trending stock tickers and (optionally) headlines/news for each ticker."""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }

    async def fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML content, ignoring cookies and increasing max header size."""
        try:
            connector = aiohttp.TCPConnector()
            cookie_jar = aiohttp.DummyCookieJar()
            async with aiohttp.ClientSession(
                headers=self.headers,
                connector=connector,
                cookie_jar=cookie_jar
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.warning(f"Failed to fetch {url} (status {response.status})")
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
        return None

    async def fetch_yahoo_trending(self) -> Set[str]:
        """Fetch trending tickers from Yahoo Finance."""
        logger.info("Fetching Yahoo Finance data...")
        url = "https://finance.yahoo.com/markets/stocks/trending"
        html = await self.fetch_html(url)
        if not html:
            return set()

        try:
            soup = BeautifulSoup(html, 'html.parser')
            tickers = {
                row.find('td', {'aria-label': 'Symbol'}).text.strip()
                for row in soup.find_all('tr')
                if row.find('td', {'aria-label': 'Symbol'})
            }
            logger.info(f"Yahoo trending tickers: {tickers}")
            return tickers
        except Exception as e:
            logger.error(f"Error parsing Yahoo trending tickers: {e}")
            return set()

    async def fetch_stocktwits_sentiment(self, limit=50) -> Set[str]:
        """Get trending stocks from Stocktwits."""
        logger.info("Fetching Stocktwits trending data...")
        try:
            url = 'https://api.stocktwits.com/api/2/trending/symbols.json'
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                logger.warning(f"Stocktwits API returned {response.status_code}")
                return set()

            data = response.json()
            symbols = data.get('symbols', [])
            if not symbols:
                return set()

            df = pd.DataFrame(symbols)
            if df.empty:
                return set()

            df = df.rename(columns={'symbol': 'ticker'})
            df = df.head(limit)
            logger.info(f"Found {len(df)} trending stocks from Stocktwits")
            return set(df['ticker'].tolist())
        except Exception as e:
            logger.error(f"Error fetching Stocktwits data: {str(e)}")
            return set()

    async def fetch_ape_wisdom(self, limit=50) -> Set[str]:
        """Get trending stocks from ApeWisdom."""
        logger.info("Fetching ApeWisdom data...")
        try:
            url = 'https://apewisdom.io/api/v1.0/filter/all-stocks/page/1'
            response = requests.get(url, headers=self.headers)
            if response.status_code != 200:
                logger.warning(f"ApeWisdom returned {response.status_code}")
                return set()

            data = response.json()
            results = data.get('results', [])
            if not results:
                return set()

            df = pd.DataFrame(results)
            if df.empty or 'ticker' not in df.columns:
                return set()

            df = df.head(limit)
            logger.info(f"Found {len(df)} trending stocks from ApeWisdom")
            return set(df['ticker'].tolist())
        except Exception as e:
            logger.error(f"Error fetching ApeWisdom data: {str(e)}")
            return set()

    async def fetch_top_25_tickers(self) -> list:
        """
        Fetches the union of tickers from:
        - Yahoo Finance 'trending'
        - Stocktwits
        - ApeWisdom
        Then returns the top 25 (sorted alphabetically).
        """
        yahoo_trending_task = asyncio.create_task(self.fetch_yahoo_trending())
        stocktwits_task = asyncio.create_task(self.fetch_stocktwits_sentiment())
        apewisdom_task = asyncio.create_task(self.fetch_ape_wisdom())

        yahoo_set,  st_set, ape_set = await asyncio.gather(
            yahoo_trending_task,
            stocktwits_task,
            apewisdom_task
        )

        all_tickers = yahoo_set.union(st_set).union(ape_set)
        sorted_list = sorted(all_tickers)
        return sorted_list[:25]

    # ------------------------------------------------------------------------
    # NEW: Fetch real headlines for each ticker from Yahoo Finance
    # ------------------------------------------------------------------------
    async def fetch_stock_news(self, ticker: str) -> List[str]:
        """
        Get recent headlines for a given ticker from Yahoo Finance's news tab.
        Returns a list of news titles (strings).
        """
        url = f"https://finance.yahoo.com/quote/{ticker}/news"
        html = await self.fetch_html(url)
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        headlines = []
        for h3 in soup.find_all('h3'):
            text = h3.get_text(strip=True)
            if text:
                headlines.append(text)

        logger.info(f"Fetched {len(headlines)} headlines for {ticker}")
        return headlines

# --------------------- Quick test of fetch_top_25_tickers --------------------
async def main():
    scraper = NewsScraper()
    top_25 = await scraper.fetch_top_25_tickers()
    logger.info(f"Top 25 Tickers: {top_25}")
    print("\nTop 25 Tickers:", top_25)

if __name__ == "__main__":
    asyncio.run(main())
