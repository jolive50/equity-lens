"""
Test script for verifying data fetching capabilities.

This script tests that we can successfully:
1. Fetch real market data from Yahoo Finance
2. Get fundamental metrics for stocks
3. Retrieve news articles
4. Run sentiment analysis with FinBERT

Run this to verify your setup is working correctly!
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Set up logging so we can see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
# This gives us access to API keys
load_dotenv()

def test_data_fetching():
    """
    Test that we can fetch real stock data.

    This function:
    1. Creates a data service (connects to Yahoo Finance)
    2. Fetches data for Apple (AAPL)
    3. Prints out what we got
    """
    print("\n" + "="*60)
    print("TEST 1: Data Fetching from Yahoo Finance")
    print("="*60 + "\n")

    try:
        # Import the data service
        # This is our main class for getting stock data
        from pipelines.realtime.sp500_data_service import get_sp500_data_service

        # Create the service instance
        # This sets up connections to Yahoo Finance
        print("📊 Creating data service...")
        service = get_sp500_data_service()
        print("✓ Data service created successfully!\n")

        # Test fetching data for Apple stock
        ticker = "AAPL"
        print(f"🔍 Fetching data for {ticker}...")

        # This actually makes the API call to Yahoo Finance
        # It will take a few seconds
        data = service.get_single_ticker_data(ticker)

        # Check what we got back
        print(f"\n✓ Successfully fetched data for {ticker}!")
        print(f"\n📈 Market Data:")
        print(f"   - {len(data['market_data'])} days of historical prices")

        # Show the most recent price
        if data['market_data']:
            latest = data['market_data'][-1]  # Get the last day
            print(f"   - Latest close: ${latest['close']}")
            print(f"   - Volume: {latest['volume']:,} shares")

        # Show fundamental metrics
        print(f"\n💼 Fundamental Metrics:")
        fundamentals = data['fundamentals']
        print(f"   - P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}")
        print(f"   - Revenue Growth: {fundamentals.get('revenue_growth', 0):.2%}")
        print(f"   - Profit Margin: {fundamentals.get('profit_margin', 0):.2%}")
        print(f"   - ROE: {fundamentals.get('roe', 0):.2%}")

        # Show news count
        print(f"\n📰 News Articles:")
        print(f"   - Found {len(data['news_data'])} recent articles")

        # Show first headline if available
        if data['news_data']:
            first_article = data['news_data'][0]
            print(f"   - Latest: {first_article['title'][:80]}...")

        return True

    except Exception as e:
        print(f"\n❌ Error fetching data: {e}")
        logger.error(f"Data fetch failed", exc_info=True)
        return False


def test_sentiment_analysis():
    """
    Test FinBERT sentiment analysis.

    This function:
    1. Creates a FinBERT sentiment analyzer
    2. Tests it on a sample news article
    3. Shows the sentiment classification
    """
    print("\n" + "="*60)
    print("TEST 2: FinBERT Sentiment Analysis")
    print("="*60 + "\n")

    try:
        # Import the sentiment analyzer
        from pipelines.realtime.sentiment.finbert import create_sentiment_analyzer

        print("🤖 Loading FinBERT model...")
        print("   (This might take 30-60 seconds the first time)")

        # Create the analyzer
        # This loads the FinBERT neural network model
        # FinBERT is specifically trained on financial text
        analyzer = create_sentiment_analyzer()
        print("✓ FinBERT model loaded!\n")

        # Create test articles with different sentiments
        test_articles = [
            {
                'title': 'Apple reports record earnings, stock surges on strong iPhone sales',
                'content': 'Apple Inc. exceeded analyst expectations with quarterly revenue of $90B.'
            },
            {
                'title': 'Tech stocks tumble as market volatility increases',
                'content': 'Major technology companies saw sharp declines amid economic uncertainty.'
            },
            {
                'title': 'Microsoft announces new product lineup',
                'content': 'The company unveiled several updates during its annual conference.'
            }
        ]

        print("📝 Analyzing test articles...\n")

        # Analyze each article
        for i, article in enumerate(test_articles, 1):
            result = analyzer.process_news_articles([article])

            print(f"Article {i}: \"{article['title'][:60]}...\"")
            print(f"   Sentiment: {result['current'].upper()}")
            print(f"   Score: {result['score']:.2%}")
            print(f"   Trend: {result['trend']}")
            print()

        print("✓ Sentiment analysis working correctly!")
        return True

    except Exception as e:
        print(f"\n❌ Error in sentiment analysis: {e}")
        logger.error(f"Sentiment analysis failed", exc_info=True)
        return False


def test_batch_analysis():
    """
    Test analyzing multiple stocks at once.

    This function:
    1. Creates a list of popular tech stocks
    2. Fetches data for all of them
    3. Shows a summary comparison
    """
    print("\n" + "="*60)
    print("TEST 3: Batch Stock Analysis")
    print("="*60 + "\n")

    try:
        from pipelines.realtime.sp500_data_service import get_sp500_data_service

        # List of stocks to analyze
        tickers = ["AAPL", "MSFT", "GOOGL"]

        print(f"📊 Analyzing {len(tickers)} stocks: {', '.join(tickers)}\n")

        service = get_sp500_data_service()

        # Fetch data for multiple tickers
        # This respects rate limits (waits between requests)
        results = service.get_multiple_tickers_data(tickers)

        print("✓ Successfully fetched data for all stocks!\n")
        print("📈 Comparison Table:")
        print("-" * 80)
        print(f"{'Ticker':<10} {'P/E Ratio':<15} {'Rev Growth':<15} {'ROE':<15} {'News':<10}")
        print("-" * 80)

        # Print comparison
        for ticker, data in results.items():
            fund = data['fundamentals']
            pe = f"{fund.get('pe_ratio', 0):.1f}" if fund.get('pe_ratio') else "N/A"
            rev = f"{fund.get('revenue_growth', 0):.2%}"
            roe = f"{fund.get('roe', 0):.2%}"
            news_count = len(data['news_data'])

            print(f"{ticker:<10} {pe:<15} {rev:<15} {roe:<15} {news_count:<10}")

        print("-" * 80)
        print("\n✓ Batch analysis complete!")
        return True

    except Exception as e:
        print(f"\n❌ Error in batch analysis: {e}")
        logger.error(f"Batch analysis failed", exc_info=True)
        return False


def test_full_workflow():
    """
    Test the complete analysis workflow.

    This simulates what happens when a user requests a stock analysis:
    1. Fetch market data
    2. Analyze sentiment
    3. Generate predictions
    4. Create explanation
    """
    print("\n" + "="*60)
    print("TEST 4: Full Analysis Workflow")
    print("="*60 + "\n")

    try:
        from pipelines.realtime.agents import (
            PredictionAgent,
            SentimentAgent,
            ExplanationAgent,
            SmartMoneyAgent,
            build_openai_llm
        )
        from pipelines.realtime.sp500_data_service import get_sp500_data_service

        # Get stock data
        ticker = "AAPL"
        print(f"🔄 Running full analysis for {ticker}...\n")

        service = get_sp500_data_service()
        data = service.get_single_ticker_data(ticker)

        print(f"✓ Step 1: Fetched market data ({len(data['market_data'])} days)")

        # Create AI agents
        # These use OpenAI's GPT model for analysis
        print(f"✓ Step 2: Creating AI agents...")
        llm = build_openai_llm("gpt-4o-mini")

        prediction_agent = PredictionAgent(llm, use_ml_model=False)  # Use LLM for demo
        sentiment_agent = SentimentAgent(llm, use_finbert=False)  # Use LLM for demo

        # Run prediction
        print(f"✓ Step 3: Running prediction analysis...")
        prediction = prediction_agent.run(
            ticker=ticker,
            market_data=data['market_data'],
            fundamentals=data['fundamentals']
        )

        print(f"   - Direction: {prediction.direction.upper()}")
        print(f"   - Confidence: {prediction.confidence:.2%}")

        # Run sentiment
        print(f"✓ Step 4: Running sentiment analysis...")
        sentiment = sentiment_agent.run(
            ticker=ticker,
            news_data=data['news_data']
        )

        print(f"   - Sentiment: {sentiment.current.upper()}")
        print(f"   - Score: {sentiment.score:.2%}")

        print(f"\n✓ Full workflow completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error in full workflow: {e}")
        logger.error(f"Full workflow failed", exc_info=True)
        return False


def main():
    """
    Main test runner.

    Runs all tests and reports results.
    """
    print("\n" + "="*60)
    print("      STOCKSENSE DATA FETCHING TEST SUITE")
    print("="*60)

    # Check for API keys first
    print("\n🔑 Checking environment variables...")
    required_keys = [
        "ALPHA_VANTAGE_API_KEY",
        "TIINGO_API_KEY",
        "FINNHUB_API_KEY",
        "NEWSAPI_KEY",
        "OPENAI_API_KEY"
    ]

    missing_keys = []
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
            print(f"   ❌ {key} not found")
        else:
            # Show first/last few characters only for security
            value = os.getenv(key)
            masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
            print(f"   ✓ {key}: {masked}")

    if missing_keys:
        print(f"\n❌ Missing API keys: {', '.join(missing_keys)}")
        print("   Please check your .env file!")
        return

    print("\n✓ All API keys found!\n")

    # Run all tests
    tests = [
        ("Data Fetching", test_data_fetching),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("Batch Analysis", test_batch_analysis),
        ("Full Workflow", test_full_workflow),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            logger.error(f"Test {name} crashed", exc_info=True)
            results.append((name, False))

    # Print summary
    print("\n" + "="*60)
    print("                    TEST SUMMARY")
    print("="*60 + "\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"   {status} - {name}")

    print(f"\n{'='*60}")
    print(f"   Total: {passed}/{total} tests passed")
    print(f"{'='*60}\n")

    if passed == total:
        print("🎉 All tests passed! Your setup is working correctly!")
        print("\nNext steps:")
        print("1. Start the backend: python -m pipelines.realtime.api")
        print("2. Start the frontend: cd frontend && npm run dev")
        print("3. Open http://localhost:3000 in your browser")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("   Common issues:")
        print("   - Missing dependencies: pip install -r requirements.txt")
        print("   - API rate limits: wait a few minutes and try again")
        print("   - Network issues: check your internet connection")


if __name__ == "__main__":
    main()
