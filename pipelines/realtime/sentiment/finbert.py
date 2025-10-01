"""FinBERT-based sentiment analysis for financial news processing."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

logger = logging.getLogger(__name__)


class FinBERTSentimentAnalyzer:
    """FinBERT-based sentiment analyzer for financial news."""
    
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """Initialize FinBERT model and tokenizer."""
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"FinBERT model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load FinBERT model: {e}")
            self.tokenizer = None
            self.model = None
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text."""
        if not self.model or not self.tokenizer:
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
        
        try:
            # Tokenize and prepare input
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
            
            # Extract probabilities for each class
            probs = probabilities.cpu().numpy()[0]
            
            # FinBERT typically has: 0=positive, 1=negative, 2=neutral
            return {
                "positive": float(probs[0]),
                "negative": float(probs[1]),
                "neutral": float(probs[2])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment of multiple texts."""
        if not self.model or not self.tokenizer:
            return [{"positive": 0.33, "negative": 0.33, "neutral": 0.34} for _ in texts]
        
        try:
            # Tokenize all texts
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
            
            # Extract probabilities for each text
            results = []
            for probs in probabilities.cpu().numpy():
                results.append({
                    "positive": float(probs[0]),
                    "negative": float(probs[1]),
                    "neutral": float(probs[2])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing batch: {e}")
            return [{"positive": 0.33, "negative": 0.33, "neutral": 0.34} for _ in texts]


class NewsSentimentProcessor:
    """Process news articles and aggregate sentiment scores."""
    
    def __init__(self, analyzer: Optional[FinBERTSentimentAnalyzer] = None):
        """Initialize with optional FinBERT analyzer."""
        self.analyzer = analyzer or FinBERTSentimentAnalyzer()
    
    def process_news_articles(self, articles: List[Dict]) -> Dict[str, any]:
        """Process a list of news articles and return aggregated sentiment."""
        if not articles:
            return {
                "current": "neutral",
                "score": 0.5,
                "trend": "stable",
                "headlines": [],
                "article_count": 0,
                "sentiment_breakdown": {"positive": 0, "negative": 0, "neutral": 0}
            }
        
        # Extract texts for analysis
        texts = []
        headlines = []
        
        for article in articles:
            # Combine title and content for better analysis
            text = f"{article.get('title', '')} {article.get('content', '')}"
            texts.append(text)
            headlines.append(article.get('title', '')[:100])  # Truncate long headlines
        
        # Analyze sentiment
        sentiment_scores = self.analyzer.analyze_batch(texts)
        
        # Aggregate results
        total_positive = sum(score["positive"] for score in sentiment_scores)
        total_negative = sum(score["negative"] for score in sentiment_scores)
        total_neutral = sum(score["neutral"] for score in sentiment_scores)
        
        article_count = len(articles)
        avg_positive = total_positive / article_count
        avg_negative = total_negative / article_count
        avg_neutral = total_neutral / article_count
        
        # Determine current sentiment
        if avg_positive > avg_negative and avg_positive > avg_neutral:
            current_sentiment = "positive"
            sentiment_score = avg_positive
        elif avg_negative > avg_positive and avg_negative > avg_neutral:
            current_sentiment = "negative"
            sentiment_score = avg_negative
        else:
            current_sentiment = "neutral"
            sentiment_score = avg_neutral
        
        # Calculate trend (simplified - would need historical data for real trend)
        trend = self._calculate_trend(sentiment_scores)
        
        return {
            "current": current_sentiment,
            "score": sentiment_score,
            "trend": trend,
            "headlines": headlines[:3],  # Top 3 headlines
            "article_count": article_count,
            "sentiment_breakdown": {
                "positive": avg_positive,
                "negative": avg_negative,
                "neutral": avg_neutral
            }
        }
    
    def _calculate_trend(self, sentiment_scores: List[Dict[str, float]]) -> str:
        """Calculate sentiment trend from recent scores."""
        if len(sentiment_scores) < 2:
            return "stable"
        
        # Simple trend calculation based on recent vs older articles
        mid_point = len(sentiment_scores) // 2
        recent_scores = sentiment_scores[:mid_point]
        older_scores = sentiment_scores[mid_point:]
        
        recent_positive = sum(score["positive"] for score in recent_scores) / len(recent_scores)
        older_positive = sum(score["positive"] for score in older_scores) / len(older_scores)
        
        if recent_positive > older_positive + 0.1:
            return "improving"
        elif recent_positive < older_positive - 0.1:
            return "declining"
        else:
            return "stable"


def create_sentiment_analyzer() -> NewsSentimentProcessor:
    """Factory function to create sentiment analyzer."""
    return NewsSentimentProcessor()


if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = create_sentiment_analyzer()
    
    test_articles = [
        {
            "title": "Apple Reports Strong Q4 Earnings",
            "content": "Apple Inc. reported better-than-expected quarterly earnings with strong iPhone sales growth."
        },
        {
            "title": "Market Volatility Concerns Investors",
            "content": "Rising inflation and interest rate concerns are causing market uncertainty."
        },
        {
            "title": "Tech Stocks Show Mixed Performance",
            "content": "Technology stocks are trading with mixed results as investors assess market conditions."
        }
    ]
    
    result = analyzer.process_news_articles(test_articles)
    print("Sentiment Analysis Result:")
    print(f"Current: {result['current']}")
    print(f"Score: {result['score']:.3f}")
    print(f"Trend: {result['trend']}")
    print(f"Headlines: {result['headlines']}")
    print(f"Breakdown: {result['sentiment_breakdown']}")