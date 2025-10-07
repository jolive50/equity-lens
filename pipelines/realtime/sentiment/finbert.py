"""FinBERT-based sentiment analysis for financial news processing.

This module uses FinBERT, a specialized AI model trained specifically for financial text,
to analyze sentiment in stock news. It's much better than general sentiment analysis
because it understands financial language and context.

Key Concepts (College Student Level):
- FinBERT: BERT model fine-tuned on financial news (understands "earnings beat" vs "earnings miss")
- Transformers: Modern AI architecture that reads text bidirectionally (looks at words before AND after)
- Sentiment: Whether news is positive (good for stock), negative (bad), or neutral
- Batch Processing: Analyzing multiple articles at once for efficiency
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# PyTorch: Deep learning framework (like TensorFlow)
# Why: FinBERT is a neural network that needs PyTorch to run
import torch

# Transformers: Hugging Face library for pre-trained AI models
# Why: Provides easy access to FinBERT and handles tokenization
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import numpy as np

logger = logging.getLogger(__name__)


class FinBERTSentimentAnalyzer:
    """FinBERT-based sentiment analyzer for financial news.

    What this does: Uses a pre-trained AI model to classify financial news as positive/negative/neutral
    Why FinBERT: It's trained on 10,000+ financial news articles, so it understands context like:
        - "Stock plunges" = negative (not about swimming)
        - "Beats estimates" = positive (better than expected)
        - "Guidance maintained" = neutral (no change)

    How it works:
    1. Load pre-trained FinBERT model from Hugging Face
    2. Convert text to tokens (numbers the model understands)
    3. Run through neural network
    4. Get probability scores for positive/negative/neutral
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """Initialize FinBERT model and tokenizer.

        What this does: Downloads and loads the FinBERT model if not already cached
        Why ProsusAI/finbert: It's one of the most popular and accurate financial sentiment models
        How it works: Hugging Face transformers library handles download and caching automatically

        Args:
            model_name: HuggingFace model identifier (ProsusAI/finbert is the default)
        """
        self.model_name = model_name

        # Determine device to run model on
        # GPU (CUDA) is much faster but not always available
        # CPU works everywhere but slower
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"FinBERT model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load FinBERT model: {e}") from e

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text.

        What this does: Takes one piece of text and returns sentiment probabilities
        Why: This is the core function that classifies text
        How it works:
        1. Tokenize text (convert to numbers)
        2. Pass through FinBERT neural network
        3. Apply softmax to get probabilities (converts raw scores to 0-1 range that sums to 1)
        4. Return probabilities for each class

        Args:
            text: The news headline and/or content to analyze
                 Example: "Apple reports record quarterly revenue"

        Returns:
            Dictionary with probabilities for each sentiment class
            Example: {"positive": 0.85, "negative": 0.05, "neutral": 0.10}
        """
        # Check if model loaded successfully
        if not self.model or not self.tokenizer:
            raise RuntimeError("FinBERT model is not available; ensure it is downloaded before running analysis.")

        try:
            # Tokenize the input text
            # What this does: Converts text to input IDs, attention masks, etc.
            # Why these parameters:
            # - return_tensors="pt": Return PyTorch tensors (not NumPy or lists)
            # - truncation=True: Cut off text longer than max_length
            # - padding=True: Pad shorter texts to uniform length
            # - max_length=512: FinBERT can handle up to 512 tokens
            inputs = self.tokenizer(
                text,
                return_tensors="pt",  # "pt" = PyTorch
                truncation=True,
                padding=True,
                max_length=512  # BERT models have 512 token limit
            ).to(self.device)  # Move to GPU if available

            # Get model predictions
            # torch.no_grad(): Disable gradient calculation (we're not training)
            # Why: Saves memory and speeds up inference
            with torch.no_grad():
                # Forward pass through the neural network
                # inputs contains: input_ids, attention_mask, token_type_ids
                outputs = self.model(**inputs)

                # outputs.logits: Raw scores from final layer (can be any number)
                # Example: [-2.1, 3.5, 0.2]
                # softmax: Converts to probabilities (0-1, sum to 1)
                # Example: [0.01, 0.94, 0.05] → 94% positive
                probabilities = torch.softmax(outputs.logits, dim=-1)

            # Extract probabilities and convert to NumPy array
            # .cpu(): Move from GPU to CPU if necessary
            # .numpy(): Convert PyTorch tensor to NumPy array
            # [0]: Get first (and only) result from batch
            probs = probabilities.cpu().numpy()[0]

            # FinBERT class mapping:
            # Index 0 = positive (good news)
            # Index 1 = negative (bad news)
            # Index 2 = neutral (factual/mixed)
            return {
                "positive": float(probs[0]),  # Convert numpy.float32 to Python float
                "negative": float(probs[1]),
                "neutral": float(probs[2])
            }

        except Exception as e:
            raise RuntimeError(f"Error analyzing text with FinBERT: {e}") from e

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment of multiple texts at once.

        What this does: Processes multiple articles in a single batch
        Why batch processing: Much faster than analyzing one-by-one
        How it works: Neural networks can process multiple inputs in parallel using GPUs

        Example:
        Input: [
            "Apple beats earnings expectations",
            "Market volatility concerns investors",
            "Company maintains guidance"
        ]
        Output: [
            {"positive": 0.89, "negative": 0.05, "neutral": 0.06},
            {"positive": 0.12, "negative": 0.72, "neutral": 0.16},
            {"positive": 0.25, "negative": 0.20, "neutral": 0.55}
        ]

        Args:
            texts: List of news texts to analyze

        Returns:
            List of sentiment probability dictionaries (one per text)
        """
        # Check if model loaded successfully
        if not self.model or not self.tokenizer:
            raise RuntimeError("FinBERT model is not available; cannot perform batch sentiment analysis.")

        try:
            # Tokenize all texts in batch
            # Same parameters as single text, but handles multiple inputs
            # The tokenizer automatically pads all texts to the same length
            inputs = self.tokenizer(
                texts,  # List of strings
                return_tensors="pt",
                truncation=True,
                padding=True,  # Pads shorter texts to match longest in batch
                max_length=512
            ).to(self.device)

            # Get model predictions for entire batch
            with torch.no_grad():
                outputs = self.model(**inputs)
                # probabilities will have shape: (batch_size, 3)
                # Example for 5 texts: (5, 3) → 5 rows, 3 columns
                probabilities = torch.softmax(outputs.logits, dim=-1)

            # Extract probabilities for each text
            results = []
            # Loop through each row (each text's probabilities)
            for probs in probabilities.cpu().numpy():
                results.append({
                    "positive": float(probs[0]),
                    "negative": float(probs[1]),
                    "neutral": float(probs[2])
                })

            return results

        except Exception as e:
            raise RuntimeError(f"Error analyzing batch with FinBERT: {e}") from e


class NewsSentimentProcessor:
    """Process news articles and aggregate sentiment scores.

    What this does: Takes raw news articles and produces actionable sentiment insights
    Why we need this: FinBERT gives us per-article sentiment, but we need overall stock sentiment
    How it works:
    1. Filter out irrelevant articles (sports, politics, etc.)
    2. Prioritize financial news (earnings, revenue, etc.)
    3. Analyze sentiment of each article with FinBERT
    4. Aggregate to get overall sentiment and trend

    Think of this as a news analyst who:
    - Reads all articles about a stock
    - Filters out noise
    - Summarizes overall market sentiment
    """

    def __init__(self, analyzer: Optional[FinBERTSentimentAnalyzer] = None):
        """Initialize with optional FinBERT analyzer.

        What this does: Creates processor with FinBERT analyzer
        Why optional: Allows dependency injection for testing
        How: Uses provided analyzer or creates new one

        Args:
            analyzer: Optional pre-configured FinBERT analyzer
        """
        # Use provided analyzer or create new one
        # This is dependency injection pattern (SOLID principle)
        self.analyzer = analyzer or FinBERTSentimentAnalyzer()

    def process_news_articles(self, articles: List[Dict]) -> Dict[str, any]:
        """Process a list of news articles and return aggregated sentiment.

        What this does: Main function that converts raw articles to sentiment summary
        Why: Users don't want to read 50 articles, they want to know: "Is sentiment positive?"
        How it works:
        1. Filter irrelevant articles
        2. Extract text from each article
        3. Run FinBERT on all articles (batch processing)
        4. Calculate averages and determine overall sentiment
        5. Detect trend (improving/declining/stable)

        Args:
            articles: List of article dictionaries
                     Each has: {title: str, content: str, timestamp: str, source: str}

        Returns:
            Sentiment summary dictionary with:
            - current: "positive"/"negative"/"neutral"
            - score: 0.0-1.0 confidence in current sentiment
            - trend: "improving"/"declining"/"stable"
            - headlines: Top 3 relevant headlines
            - article_count: How many articles analyzed
            - sentiment_breakdown: Average probabilities
        """
        # Handle empty input
        if not articles:
            raise ValueError("At least one article is required for FinBERT sentiment analysis.")

        # Filter articles to keep only financially relevant ones
        # Why: "Apple CEO plays golf" is less relevant than "Apple misses revenue targets"
        articles = self._filter_relevant_articles(articles)

        # Extract texts for analysis
        texts = []  # Will hold combined title+content for each article
        headlines = []  # Will hold just titles for display

        for article in articles:
            # Combine title and content for more context
            # Why: Title might be "Apple Announces" but content explains what they announced
            text = f"{article.get('title', '')} {article.get('content', '')}"
            texts.append(text)

            # Store headline (truncate to 100 chars for display)
            headlines.append(article.get('title', '')[:100])

        if not texts:
            raise RuntimeError("No financially relevant articles available for sentiment analysis.")

        # Analyze sentiment of all articles at once (batch processing)
        # This is much faster than analyzing one-by-one
        sentiment_scores = self.analyzer.analyze_batch(texts)

        # Aggregate results across all articles
        # Calculate total probabilities (we'll divide by count to get averages)
        total_positive = sum(score["positive"] for score in sentiment_scores)
        total_negative = sum(score["negative"] for score in sentiment_scores)
        total_neutral = sum(score["neutral"] for score in sentiment_scores)

        # Calculate averages
        article_count = len(articles)
        avg_positive = total_positive / article_count
        avg_negative = total_negative / article_count
        avg_neutral = total_neutral / article_count

        # Determine current overall sentiment
        # Whichever has highest average probability wins
        if avg_positive > avg_negative and avg_positive > avg_neutral:
            current_sentiment = "positive"
            sentiment_score = avg_positive
        elif avg_negative > avg_positive and avg_negative > avg_neutral:
            current_sentiment = "negative"
            sentiment_score = avg_negative
        else:
            current_sentiment = "neutral"
            sentiment_score = avg_neutral

        # Calculate sentiment trend
        # Are things getting better or worse over time?
        trend = self._calculate_trend(sentiment_scores)

        return {
            "current": current_sentiment,
            "score": sentiment_score,
            "trend": trend,
            "headlines": headlines[:3],  # Show top 3 most relevant headlines
            "article_count": article_count,
            "sentiment_breakdown": {
                "positive": avg_positive,
                "negative": avg_negative,
                "neutral": avg_neutral
            }
        }

    def _filter_relevant_articles(self, articles: List[Dict]) -> List[Dict]:
        """Filter articles to prioritize financial relevance and recency.

        What this does: Scores each article and keeps only the most relevant ones
        Why: Not all news is equally important for stock analysis
        How it works:
        1. Score each article based on keywords
        2. Give bonus points for financial terms (earnings, revenue, etc.)
        3. Give penalty points for non-financial content (sports, weather)
        4. Keep top 15 articles

        Args:
            articles: All available articles

        Returns:
            Filtered list of most relevant articles (max 15)
        """
        # Score each article for financial relevance
        scored_articles = []

        for article in articles:
            score = 0  # Start with 0 points

            # Extract title and content, convert to lowercase for matching
            title = article.get("title", "").lower()
            content = article.get("content", "").lower()

            # Financial relevance keywords
            # These words indicate the article is about business/finance
            financial_keywords = [
                "earnings", "revenue", "profit", "growth", "financial", "dividend",
                "stock", "market", "investor", "quarterly", "guidance", "forecast",
                "acquisition", "merger", "partnership", "product launch", "innovation",
                "sales", "margin", "ebitda", "cash flow", "debt", "equity"
            ]

            # Score based on keyword presence
            for keyword in financial_keywords:
                if keyword in title:
                    # Keywords in title are more important
                    score += 3
                elif keyword in content:
                    # Keywords in content still valuable but less so
                    score += 1

            # Bonus for earnings-related news (most important financial event)
            # Why: Quarterly earnings reports are major stock price drivers
            earnings_keywords = ["earnings", "profit", "revenue", "quarterly"]
            if any(keyword in title for keyword in earnings_keywords):
                score += 2  # Extra bonus points

            # Penalty for non-financial content
            # These topics are usually irrelevant to stock analysis
            non_financial_keywords = ["politics", "sports", "entertainment", "weather", "celebrity"]
            if any(keyword in title for keyword in non_financial_keywords):
                score -= 5  # Deduct points

            # Only keep articles with positive scores
            if score >= 0:
                scored_articles.append((score, article))

        # Sort by score (highest first) and take top articles
        scored_articles.sort(reverse=True, key=lambda x: x[0])

        # Return top 15 most relevant articles
        # Why 15: Balance between having enough data and avoiding noise
        return [article for _, article in scored_articles[:15]]

    def analyze_sector_sentiment(self, ticker_news_map: Dict[str, List[Dict]]) -> Dict[str, any]:
        """Analyze sentiment across multiple S&P 500 tickers.

        What this does: Analyzes sentiment for multiple stocks at once to get sector view
        Why: Helpful for portfolio analysis or sector-wide trends
        How it works:
        1. Analyze sentiment for each ticker individually
        2. Aggregate to get sector-wide sentiment
        3. Calculate sentiment convergence (how much tickers agree)

        Example:
        Input: {
            "AAPL": [article1, article2, ...],
            "MSFT": [article3, article4, ...],
            "GOOGL": [article5, article6, ...]
        }
        Output: {
            "sector_sentiment": {
                "AAPL": {"current": "positive", ...},
                "MSFT": {"current": "positive", ...},
                "GOOGL": {"current": "neutral", ...}
            },
            "overall_sentiment": "positive",
            "convergence_score": 0.67  # 2 out of 3 agree
        }

        Args:
            ticker_news_map: Dictionary mapping ticker symbols to their news articles

        Returns:
            Sector-wide sentiment analysis including individual results and aggregates
        """
        sector_results = {}

        # Analyze sentiment for each ticker
        for ticker, articles in ticker_news_map.items():
            logger.info(f"Analyzing sentiment for {ticker} ({len(articles)} articles)")
            sector_results[ticker] = self.process_news_articles(articles)

        # Calculate sector-wide aggregated sentiment
        # We weight each ticker by its article count (more articles = more data = more weight)
        total_positive = 0
        total_negative = 0
        total_neutral = 0
        total_articles = 0

        for result in sector_results.values():
            breakdown = result["sentiment_breakdown"]
            article_count = result["article_count"]

            # Accumulate weighted sentiment scores
            total_positive += breakdown["positive"] * article_count
            total_negative += breakdown["negative"] * article_count
            total_neutral += breakdown["neutral"] * article_count
            total_articles += article_count

        # Handle case where no articles found
        if total_articles == 0:
            logger.warning("No articles found for any ticker in sector analysis")
            return {
                "sector_sentiment": sector_results,
                "overall_sentiment": "neutral",
                "overall_score": 0.5,
                "convergence_score": 0.5,
                "total_articles": 0
            }

        # Calculate sector-wide averages
        avg_positive = total_positive / total_articles
        avg_negative = total_negative / total_articles
        avg_neutral = total_neutral / total_articles

        # Determine overall sector sentiment
        if avg_positive > max(avg_negative, avg_neutral):
            overall_sentiment = "positive"
            overall_score = avg_positive
        elif avg_negative > max(avg_positive, avg_neutral):
            overall_sentiment = "negative"
            overall_score = avg_negative
        else:
            overall_sentiment = "neutral"
            overall_score = avg_neutral

        # Calculate sentiment convergence
        # This measures how much the different stocks agree on sentiment
        # High convergence = sector-wide trend, Low convergence = mixed signals
        sentiment_consistency = self._calculate_sentiment_consistency(sector_results)

        return {
            "sector_sentiment": sector_results,  # Individual ticker results
            "overall_sentiment": overall_sentiment,  # Aggregated sentiment
            "overall_score": overall_score,  # Confidence in overall sentiment
            "convergence_score": sentiment_consistency,  # How much tickers agree
            "total_articles": total_articles  # Total articles analyzed
        }

    def _calculate_sentiment_consistency(self, sector_results: Dict[str, Dict]) -> float:
        """Calculate how consistent sentiment is across different tickers.

        What this does: Measures agreement between different stocks' sentiment
        Why: High agreement suggests sector-wide trend, disagreement suggests mixed signals
        How it works: Calculates proportion of stocks with the same sentiment

        Example:
        - 5 stocks, all positive → consistency = 1.0 (100% agreement)
        - 5 stocks, 3 positive, 2 negative → consistency = 0.6 (60% agreement)
        - 5 stocks, all different → consistency = 0.4 (40% agreement)

        Args:
            sector_results: Dictionary of per-ticker sentiment results

        Returns:
            Consistency score from 0.0 to 1.0
        """
        # Need at least 2 stocks to measure consistency
        if len(sector_results) < 2:
            return 1.0  # Single stock is perfectly consistent with itself

        # Extract sentiment labels for all tickers
        sentiments = [result["current"] for result in sector_results.values()]

        # Count occurrences of each sentiment
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for sentiment in sentiments:
            sentiment_counts[sentiment] += 1

        # Calculate consistency as proportion of most common sentiment
        total = len(sentiments)
        max_count = max(sentiment_counts.values())

        # Example: If 8 out of 10 stocks are positive, consistency = 0.8
        consistency = max_count / total

        # Clamp to [0.0, 1.0] range (should already be in range, but just in case)
        return min(1.0, consistency)

    def _calculate_trend(self, sentiment_scores: List[Dict[str, float]]) -> str:
        """Calculate sentiment trend from recent scores.

        What this does: Determines if sentiment is improving, declining, or stable
        Why: Important to know if things are getting better or worse
        How it works: Compare recent articles to older articles

        Limitations: This is simplified - real trend analysis would need timestamps

        Args:
            sentiment_scores: List of sentiment scores (ordered by article order)

        Returns:
            "improving", "declining", or "stable"
        """
        # Need at least 2 articles to detect trend
        if len(sentiment_scores) < 2:
            return "stable"  # Not enough data to detect trend

        # Split articles into recent vs older
        # Assumption: Articles are ordered chronologically (most recent first)
        mid_point = len(sentiment_scores) // 2
        recent_scores = sentiment_scores[:mid_point]  # First half (more recent)
        older_scores = sentiment_scores[mid_point:]   # Second half (older)

        # Calculate average positive sentiment for each group
        recent_positive = sum(score["positive"] for score in recent_scores) / len(recent_scores)
        older_positive = sum(score["positive"] for score in older_scores) / len(older_scores)

        # Determine trend based on change
        # Use 0.1 threshold to avoid classifying small fluctuations as trends
        if recent_positive > older_positive + 0.1:
            return "improving"  # Recent sentiment significantly more positive
        elif recent_positive < older_positive - 0.1:
            return "declining"  # Recent sentiment significantly more negative
        else:
            return "stable"  # No significant change


def create_sentiment_analyzer() -> NewsSentimentProcessor:
    """Factory function to create sentiment analyzer.

    What this does: Creates and returns a new sentiment processor
    Why factory pattern: Centralizes object creation, makes testing easier
    How: Simply instantiates NewsSentimentProcessor (which internally creates FinBERT)

    Returns:
        Configured NewsSentimentProcessor ready to use
    """
    return NewsSentimentProcessor()


# Test code (runs when this file is executed directly)
if __name__ == "__main__":
    print("Testing FinBERT Sentiment Analyzer...")
    print("Note: First run will download the model (~400MB), subsequent runs use cached version\n")

    # Create analyzer
    analyzer = create_sentiment_analyzer()

    # Test articles with different sentiments
    test_articles = [
        {
            "title": "Apple Reports Strong Q4 Earnings, Beats Expectations",
            "content": "Apple Inc. reported better-than-expected quarterly earnings with strong iPhone sales growth and improved margins."
        },
        {
            "title": "Market Volatility Concerns Investors Amid Economic Uncertainty",
            "content": "Rising inflation and interest rate concerns are causing market uncertainty and investor anxiety."
        },
        {
            "title": "Tech Stocks Show Mixed Performance in Trading Session",
            "content": "Technology stocks are trading with mixed results as investors assess market conditions and company fundamentals."
        },
        {
            "title": "Company Announces New Product Launch for Q1",
            "content": "The company revealed plans to launch innovative products next quarter, maintaining current guidance."
        }
    ]

    # Analyze sentiment
    print("Analyzing sentiment of test articles...\n")
    result = analyzer.process_news_articles(test_articles)

    # Display results
    print("=" * 60)
    print("SENTIMENT ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Overall Sentiment: {result['current'].upper()}")
    print(f"Confidence Score: {result['score']:.1%}")
    print(f"Trend: {result['trend'].capitalize()}")
    print(f"Articles Analyzed: {result['article_count']}")
    print(f"\nSentiment Breakdown:")
    print(f"  Positive: {result['sentiment_breakdown']['positive']:.1%}")
    print(f"  Negative: {result['sentiment_breakdown']['negative']:.1%}")
    print(f"  Neutral:  {result['sentiment_breakdown']['neutral']:.1%}")
    print(f"\nTop Headlines:")
    for i, headline in enumerate(result['headlines'], 1):
        print(f"  {i}. {headline}")
    print("=" * 60)

    print("\nSentiment analyzer test complete!")
