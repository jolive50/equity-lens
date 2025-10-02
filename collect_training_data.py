"""
Historical Data Collection Script for Model Training

This script collects historical stock data and saves it for training our ML models.

What it does:
1. Fetches 1+ years of historical data for major S&P 500 stocks
2. Gets fundamental metrics over time
3. Collects historical news and sentiment
4. Saves everything in a structured format for training

Run this before training models!
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

# Set up detailed logging so we can track progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load API keys from .env
load_dotenv()


def create_data_directories():
    """
    Create folder structure for storing training data.

    Creates:
    - data/training/market_data/    <- Historical prices
    - data/training/fundamentals/   <- Financial metrics
    - data/training/sentiment/      <- News sentiment scores
    - data/training/labels/         <- Target labels (up/down/neutral)
    """
    base_dir = Path("data/training")
    subdirs = ["market_data", "fundamentals", "sentiment", "labels"]

    for subdir in subdirs:
        path = base_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created directory: {path}")

    return base_dir


def collect_historical_data(ticker: str, period: str = "2y") -> Dict:
    """
    Fetch historical data for a single stock.

    Args:
        ticker: Stock symbol (e.g., "AAPL")
        period: How far back to go ("1y", "2y", "5y", "max")

    Returns:
        Dictionary with market_data, fundamentals, and metadata

    This uses yfinance to get REAL historical data, not fake data!
    """
    logger.info(f"📊 Fetching {period} of historical data for {ticker}...")

    try:
        import yfinance as yf

        # Create a Ticker object - this connects to Yahoo Finance
        stock = yf.Ticker(ticker)

        # Get historical price data
        # This returns a pandas DataFrame with OHLCV data
        hist = stock.history(period=period)

        if hist.empty:
            logger.warning(f"No data returned for {ticker}")
            return None

        # Convert DataFrame to list of dictionaries for easy JSON storage
        market_data = []
        for date, row in hist.iterrows():
            market_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
            })

        # Get fundamental metrics
        info = stock.info
        fundamentals = {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "revenue_growth": info.get("revenueGrowth", None),
            "profit_margin": info.get("profitMargins", None),
            "debt_to_equity": info.get("debtToEquity", None),
            "roe": info.get("returnOnEquity", None),
            "beta": info.get("beta", None),
        }

        logger.info(f"✓ Got {len(market_data)} days of data for {ticker}")

        return {
            "ticker": ticker,
            "market_data": market_data,
            "fundamentals": fundamentals,
            "collected_at": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")
        return None


def calculate_labels(market_data: List[Dict], lookahead_days: int = 5) -> List[Dict]:
    """
    Create training labels from historical data.

    For each day, we look ahead 'lookahead_days' and determine:
    - "up" if price increased by > 2%
    - "down" if price decreased by > 2%
    - "neutral" otherwise

    Args:
        market_data: List of daily price data
        lookahead_days: How many days ahead to look (default: 5 for 1 week)

    Returns:
        List of labels with dates and classifications
    """
    labels = []

    # We can't label the last few days (no future data to look at)
    for i in range(len(market_data) - lookahead_days):
        current_day = market_data[i]
        future_day = market_data[i + lookahead_days]

        # Calculate price change percentage
        current_price = current_day["close"]
        future_price = future_day["close"]
        price_change_pct = (future_price - current_price) / current_price

        # Classify the movement
        if price_change_pct > 0.02:  # More than 2% increase
            direction = "up"
        elif price_change_pct < -0.02:  # More than 2% decrease
            direction = "down"
        else:  # Between -2% and +2%
            direction = "neutral"

        labels.append({
            "date": current_day["date"],
            "direction": direction,
            "price_change_pct": round(price_change_pct, 4),
            "current_price": current_price,
            "future_price": future_price,
            "lookahead_days": lookahead_days
        })

    logger.info(f"✓ Created {len(labels)} training labels")

    # Show distribution
    up_count = sum(1 for l in labels if l["direction"] == "up")
    down_count = sum(1 for l in labels if l["direction"] == "down")
    neutral_count = sum(1 for l in labels if l["direction"] == "neutral")

    logger.info(f"   Distribution: {up_count} up, {down_count} down, {neutral_count} neutral")

    return labels


def save_training_data(data: Dict, base_dir: Path):
    """
    Save collected data to disk in JSON format.

    Organizes data by ticker:
    - data/training/market_data/AAPL.json
    - data/training/fundamentals/AAPL.json
    - data/training/labels/AAPL.json
    """
    ticker = data["ticker"]

    # Save market data (prices)
    market_file = base_dir / "market_data" / f"{ticker}.json"
    with open(market_file, 'w') as f:
        json.dump(data["market_data"], f, indent=2)
    logger.info(f"✓ Saved market data to {market_file}")

    # Save fundamentals
    fund_file = base_dir / "fundamentals" / f"{ticker}.json"
    with open(fund_file, 'w') as f:
        json.dump(data["fundamentals"], f, indent=2)
    logger.info(f"✓ Saved fundamentals to {fund_file}")

    # Calculate and save labels
    labels = calculate_labels(data["market_data"], lookahead_days=5)
    labels_file = base_dir / "labels" / f"{ticker}.json"
    with open(labels_file, 'w') as f:
        json.dump(labels, f, indent=2)
    logger.info(f"✓ Saved labels to {labels_file}")


def main():
    """
    Main function to collect training data for multiple stocks.
    """
    print("\n" + "="*70)
    print("       STOCKSENSE HISTORICAL DATA COLLECTION")
    print("="*70 + "\n")

    # List of S&P 500 stocks to collect data for
    # Start with major tech stocks, then expand
    tickers = [
        # Tech giants
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Google
        "AMZN",  # Amazon
        "META",  # Meta/Facebook
        "NVDA",  # Nvidia
        "TSLA",  # Tesla

        # Other major companies
        "JPM",   # JP Morgan
        "V",     # Visa
        "WMT",   # Walmart
        "JNJ",   # Johnson & Johnson
        "PG",    # Procter & Gamble
    ]

    print(f"📋 Will collect data for {len(tickers)} stocks:")
    print(f"   {', '.join(tickers)}\n")

    # Create directory structure
    print("📁 Setting up directories...")
    base_dir = create_data_directories()
    print()

    # Collect data for each stock
    successful = 0
    failed = 0

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
        print("-" * 60)

        try:
            # Fetch historical data (2 years)
            data = collect_historical_data(ticker, period="2y")

            if data:
                # Save to disk
                save_training_data(data, base_dir)
                successful += 1
                print(f"✓ {ticker} complete!")
            else:
                failed += 1
                print(f"✗ {ticker} failed - no data returned")

            # Be nice to the API - wait between requests
            if i < len(tickers):
                import time
                print("⏳ Waiting 2 seconds before next request...")
                time.sleep(2)

        except Exception as e:
            failed += 1
            logger.error(f"Error processing {ticker}: {e}")
            print(f"✗ {ticker} failed - {str(e)[:50]}")

    # Print summary
    print("\n" + "="*70)
    print("                        SUMMARY")
    print("="*70)
    print(f"\n✓ Successfully collected: {successful}/{len(tickers)} stocks")
    print(f"✗ Failed: {failed}/{len(tickers)} stocks")
    print(f"\n📁 Data saved to: {base_dir.absolute()}")

    print("\n📊 Training Data Statistics:")
    print(f"   - Market data files: {len(list((base_dir / 'market_data').glob('*.json')))}")
    print(f"   - Fundamental files: {len(list((base_dir / 'fundamentals').glob('*.json')))}")
    print(f"   - Label files: {len(list((base_dir / 'labels').glob('*.json')))}")

    print("\n🎯 Next steps:")
    print("   1. Review the collected data in data/training/")
    print("   2. Run: python train_model.py")
    print("   3. This will train an ML model on the historical data")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
