import os
import re
import logging
import nltk
import spacy
import pandas as pd
import numpy as np
import asyncio

from datetime import datetime
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from typing import Dict, List, Optional, Union, Tuple
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Import the updated NewsScraper
from data.news_scraper import NewsScraper

# -----------------------------------------------------------------------------
# Setup logging
# -----------------------------------------------------------------------------
os.makedirs("data/logs", exist_ok=True)
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"data/logs/{current_time_str}_sentiment.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# SentimentAnalyzer Class
# -----------------------------------------------------------------------------
class SentimentAnalyzer:
    """Advanced sentiment analysis for financial texts."""
    
    def __init__(self, use_transformers: bool = False):
        # Initialize NLTK
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            self.sia = SentimentIntensityAnalyzer()
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.error(f"Error initializing NLTK: {str(e)}")
            raise

        # Initialize spaCy
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            raise

        self.use_transformers = use_transformers
        if use_transformers:
            try:
                self.transformer = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    return_all_scores=True
                )
            except Exception as e:
                logger.warning(f"Error loading transformer: {str(e)}")
                self.use_transformers = False

        self.financial_words = self._load_financial_lexicon()
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def _load_financial_lexicon(self) -> Dict[str, float]:
        return {
            "bullish": 0.8,
            "bearish": -0.8,
            "outperform": 0.6,
            "underperform": -0.6,
            "upgrade": 0.7,
            "downgrade": -0.7,
            "buy": 0.5,
            "sell": -0.5,
            "hold": 0.0,
            "overweight": 0.4,
            "underweight": -0.4,
            "positive": 0.3,
            "negative": -0.3,
            "beat": 0.6,
            "miss": -0.6,
            "above": 0.4,
            "below": -0.4,
            "growth": 0.5,
            "decline": -0.5,
            "strong": 0.4,
            "weak": -0.4
        }

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'(?<!\$)\b\d+\b(?!\%)', '', text)
        text = ' '.join(text.split())
        return text

    def extract_financial_metrics(self, text: str) -> Dict[str, List[Tuple[str, float]]]:
        metrics = defaultdict(list)
        doc = self.nlp(text)
        
        money_pattern = r'\$?\d+\.?\d*[MBK]?'
        percentage_pattern = r'\d+\.?\d*\%'
        
        for entity in doc.ents:
            if entity.label_ in ['MONEY', 'PERCENT', 'QUANTITY']:
                # Money
                money = re.findall(money_pattern, entity.text)
                if money:
                    try:
                        numeric_val = re.sub(r'[^\d\.]', '', money[0])
                        metrics['monetary'].append((entity.text, float(numeric_val)))
                    except ValueError:
                        pass
                # Percentage
                perc = re.findall(percentage_pattern, entity.text)
                if perc:
                    try:
                        numeric_val = perc[0].replace('%', '')
                        metrics['percentage'].append((entity.text, float(numeric_val)))
                    except ValueError:
                        pass
        return dict(metrics)

    def analyze_sentiment(
        self, 
        text: str, 
        method: str = 'ensemble', 
        source_reliability: float = 1.0
    ) -> Dict[str, Union[float, str, Dict]]:
        """Analyze sentiment for a single text."""
        source_reliability = max(0.0, min(1.0, source_reliability))
        try:
            preprocessed = self.preprocess_text(text)
            
            results = {
                'original_text': text,
                'preprocessed_text': preprocessed,
                'financial_metrics': self.extract_financial_metrics(text)
            }

            # VADER
            vader = self.sia.polarity_scores(preprocessed)
            results['vader'] = vader

            # TextBlob
            blob = TextBlob(preprocessed)
            results['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }

            # Transformer (FinBERT) if enabled
            if self.use_transformers:
                t_result = self.transformer(preprocessed)
                if t_result:
                    scores_list = t_result[0]
                    results['transformer'] = {'scores': scores_list}
                    max_label = max(scores_list, key=lambda x: x['score'])
                    results['transformer']['final_label'] = max_label['label']
                    results['transformer']['final_score'] = max_label['score']

            # Financial lexicon
            fin_lex = self._calculate_financial_sentiment(preprocessed)
            results['financial'] = fin_lex

            # Ensemble or single method
            if method == 'ensemble':
                ensemble_score = self._calculate_ensemble_sentiment(results)
                ensemble_score *= source_reliability
                results['ensemble'] = ensemble_score
                results['final_sentiment'] = ensemble_score
            else:
                if method == 'vader':
                    results['final_sentiment'] = vader['compound'] * source_reliability
                elif method == 'textblob':
                    results['final_sentiment'] = blob.sentiment.polarity * source_reliability
                elif method == 'transformer' and self.use_transformers:
                    label = results['transformer']['final_label']
                    label_score = (1.0 if label == 'POSITIVE' 
                                   else -1.0 if label == 'NEGATIVE' 
                                   else 0.0)
                    results['final_sentiment'] = label_score * results['transformer']['final_score'] * source_reliability
                else:
                    results['final_sentiment'] = vader['compound'] * source_reliability

            # Confidence
            results['confidence'] = self._calculate_confidence(results)

            # Log the single-text result
            logger.info(
                f"Analyzed: '{text[:40]}...' -> "
                f"sentiment={results['final_sentiment']:.3f}, "
                f"confidence={results['confidence']:.2f}"
            )
            
            return results

        except Exception as e:
            logger.error(f"Error in analyze_sentiment: {str(e)}")
            return {}

    def _calculate_financial_sentiment(self, text: str) -> Dict[str, float]:
        words = word_tokenize(text)
        score = 0
        relevant_words = 0
        for w in words:
            if w in self.financial_words:
                score += self.financial_words[w]
                relevant_words += 1

        if relevant_words == 0:
            return {'score': 0.0, 'confidence': 0.0}

        avg_score = score / relevant_words
        return {'score': avg_score, 'confidence': min(1.0, relevant_words / 10)}

    def _calculate_ensemble_sentiment(self, results: Dict) -> float:
        weights = {
            'vader': 0.3,
            'textblob': 0.2,
            'financial': 0.3,
            'transformer': 0.2
        }
        
        sentiment_scores = []
        total_weight = 0.0

        # VADER
        sentiment_scores.append((results['vader']['compound'], weights['vader']))
        total_weight += weights['vader']

        # TextBlob
        sentiment_scores.append((results['textblob']['polarity'], weights['textblob']))
        total_weight += weights['textblob']

        # Financial
        sentiment_scores.append((results['financial']['score'], weights['financial']))
        total_weight += weights['financial']

        # Transformer (if used)
        if self.use_transformers and 'transformer' in results:
            label = results['transformer']['final_label']
            label_score = (1.0 if label == 'POSITIVE' else
                           -1.0 if label == 'NEGATIVE' else 0.0)
            confidence = results['transformer']['final_score']
            sentiment_scores.append((label_score * confidence, weights['transformer']))
            total_weight += weights['transformer']

        if total_weight == 0:
            return 0.0

        weighted_sum = sum(score * w for score, w in sentiment_scores)
        return weighted_sum / total_weight

    def _calculate_confidence(self, results: Dict) -> float:
        confs = []
        confs.append(abs(results['vader']['compound']))
        confs.append(1 - results['textblob']['subjectivity'])
        confs.append(results['financial']['confidence'])
        if self.use_transformers and 'transformer' in results:
            confs.append(results['transformer']['final_score'])

        if not confs:
            return 0.0
        return sum(confs) / len(confs)

    def analyze_text_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze a list of texts (headlines)."""
        return [self.analyze_sentiment(t, method='ensemble') for t in texts]

    def aggregate_sentiment_scores(
        self, 
        sentiment_results: List[Dict[str, Union[float, str, Dict]]]
    ) -> Dict[str, Union[float, str]]:
        if not sentiment_results:
            return {'aggregated_score': 0.0, 'label': 'Unknown', 'num_sources': 0}

        scores = [res.get('final_sentiment', 0.0) for res in sentiment_results]
        aggregated_score = np.mean(scores) if scores else 0.0
        label = self.label_sentiment_score(aggregated_score)
        
        logger.info(
            f"Aggregated {len(sentiment_results)} headlines => "
            f"score={aggregated_score:.3f}, label={label}"
        )
        return {
            'aggregated_score': aggregated_score,
            'label': label,
            'num_sources': len(sentiment_results)
        }

    def label_sentiment_score(self, score: float) -> str:
        if score >= 0.6:
            return "Strong Buy"
        elif score >= 0.3:
            return "Buy"
        elif score <= -0.6:
            return "Strong Sell"
        elif score <= -0.3:
            return "Sell"
        else:
            return "Hold"

# -----------------------------------------------------------------------------
# Main Async Function: pull top 25 tickers, get real headlines, analyze them
# -----------------------------------------------------------------------------
async def run_analysis():
    # 1) Fetch the top 25 trending tickers
    scraper = NewsScraper()
    top_25_tickers = await scraper.fetch_top_25_tickers()

    logger.info(f"\nTop 25 Tickers: {top_25_tickers}\n")
    if not top_25_tickers:
        logger.warning("No tickers found.")
        return

    # 2) Create the sentiment analyzer
    analyzer = SentimentAnalyzer(use_transformers=False)

    # 3) For each ticker, fetch real Yahoo Finance news headlines
    for ticker in top_25_tickers:
        headlines = await scraper.fetch_stock_news(ticker)
        if not headlines:
            logger.info(f"No headlines found for {ticker}, skipping.")
            continue

        # 4) Analyze each headline
        sentiment_results = analyzer.analyze_text_batch(headlines)

        # 5) Aggregate final sentiment
        agg = analyzer.aggregate_sentiment_scores(sentiment_results)

        logger.info(f"{ticker} => Score: {agg['aggregated_score']:.3f}, Label: {agg['label']}")
        print(f"{ticker} => {agg}")

# -----------------------------------------------------------------------------
# If called directly, run the analysis
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(run_analysis())
