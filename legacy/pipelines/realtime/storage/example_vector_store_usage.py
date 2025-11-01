"""Example usage script for ChromaDB VectorStore.

This script demonstrates how to use the VectorStore for:
- Adding news articles with embeddings
- Searching for similar news
- Adding historical analyses
- Finding similar past analyses

WHAT: Practical examples of VectorStore API usage
WHY: Help team members understand how to integrate semantic search
HOW: Runnable script with real data examples
DATA: Sample news/analyses → vector operations → printed results
"""
from pipelines.realtime.storage.vector_store import VectorStore


def main():
    """Demonstrate VectorStore usage with examples."""

    print("=" * 70)
    print("ChromaDB VectorStore Example Usage")
    print("=" * 70)

    # ===== STEP 1: Initialize VectorStore =====
    print("\n1. Initializing VectorStore...")
    store = VectorStore(
        persist_directory="data/chroma",
        reset_on_init=True  # Start fresh for demo
    )
    print("   [OK] VectorStore initialized")

    # ===== STEP 2: Add News Articles =====
    print("\n2. Adding news articles...")

    apple_news = [
        {
            "headline": "Apple Reports Record Q4 Earnings",
            "content": "Apple Inc. announced record fourth quarter earnings today, "
                      "beating analyst expectations on strong iPhone 15 sales and "
                      "services revenue growth.",
            "date": "2025-10-28",
            "source": "Reuters",
            "sentiment": 0.85,
        },
        {
            "headline": "Apple Stock Rises 5% After Earnings Beat",
            "content": "Shares of Apple rose 5% in after-hours trading following "
                      "the company's better-than-expected quarterly results. "
                      "Revenue from services segment exceeded projections.",
            "date": "2025-10-28",
            "source": "Bloomberg",
            "sentiment": 0.80,
        },
        {
            "headline": "Apple Announces New AI Features for iPhone",
            "content": "Apple unveiled new artificial intelligence features for "
                      "the iPhone during its product launch event, focusing on "
                      "on-device processing and privacy.",
            "date": "2025-10-27",
            "source": "TechCrunch",
            "sentiment": 0.70,
        },
    ]

    count = store.add_news_articles(ticker="AAPL", articles=apple_news)
    print(f"   [OK] Added {count} news articles for AAPL")

    # Add some Microsoft news for comparison
    msft_news = [
        {
            "headline": "Microsoft Cloud Revenue Surges",
            "content": "Microsoft reported strong Azure cloud growth this quarter, "
                      "with enterprise customers increasing their usage.",
            "date": "2025-10-27",
            "source": "WSJ",
            "sentiment": 0.75,
        },
    ]

    count = store.add_news_articles(ticker="MSFT", articles=msft_news)
    print(f"   [OK] Added {count} news articles for MSFT")

    # ===== STEP 3: Search for Similar News =====
    print("\n3. Searching for similar news...")

    # Example 1: Find earnings-related news
    print("\n   Query: 'company reports strong financial results'")
    results = store.search_similar_news(
        query="company reports strong financial results",
        ticker="AAPL",  # Filter to Apple news only
        n_results=2
    )

    print(f"   Found {len(results)} similar articles:")
    for i, result in enumerate(results, 1):
        print(f"\n   Result {i}:")
        print(f"   - Headline: {result['headline']}")
        print(f"   - Source: {result['source']} ({result['date']})")
        print(f"   - Distance: {result['distance']:.3f} (lower = more similar)")
        print(f"   - Sentiment: {result['sentiment']:.2f}")

    # Example 2: Find AI-related news
    print("\n   Query: 'artificial intelligence product announcement'")
    results = store.search_similar_news(
        query="artificial intelligence product announcement",
        ticker="AAPL",
        n_results=2
    )

    print(f"   Found {len(results)} similar articles:")
    for i, result in enumerate(results, 1):
        print(f"\n   Result {i}:")
        print(f"   - Headline: {result['headline']}")
        print(f"   - Distance: {result['distance']:.3f}")

    # Example 3: Cross-ticker search (no ticker filter)
    print("\n   Query: 'tech company cloud services growth'")
    results = store.search_similar_news(
        query="tech company cloud services growth",
        n_results=3  # Search across all tickers
    )

    print(f"   Found {len(results)} similar articles:")
    for i, result in enumerate(results, 1):
        print(f"\n   Result {i}:")
        print(f"   - Ticker: {result['ticker']}")
        print(f"   - Headline: {result['headline']}")
        print(f"   - Distance: {result['distance']:.3f}")

    # ===== STEP 4: Add Historical Analyses =====
    print("\n4. Adding historical analyses...")

    analysis_text = (
        "Strong bullish sentiment detected with positive earnings surprise. "
        "Technical indicators show upward momentum with increasing volume. "
        "Institutional buying detected in recent 13F filings. "
        "High confidence prediction for upward movement over next 30 days."
    )

    analysis_metadata = {
        "date": "2025-10-15",
        "prediction": "up",
        "confidence": 0.88,
        "sentiment": "positive",
        "actual_outcome": "up",  # Recorded after 30 days
    }

    success = store.add_analysis(
        ticker="AAPL",
        analysis_text=analysis_text,
        metadata=analysis_metadata
    )

    print(f"   [OK] Added analysis for AAPL")

    # ===== STEP 5: Search for Similar Analyses =====
    print("\n5. Searching for similar historical analyses...")

    current_situation = (
        "Bullish sentiment with strong institutional activity "
        "and positive technical momentum"
    )

    print(f"\n   Current situation: '{current_situation}'")

    similar_analyses = store.search_similar_analyses(
        query=current_situation,
        ticker="AAPL",
        n_results=3
    )

    print(f"   Found {len(similar_analyses)} similar past analyses:")
    for i, analysis in enumerate(similar_analyses, 1):
        print(f"\n   Analysis {i}:")
        print(f"   - Date: {analysis['date']}")
        print(f"   - Prediction: {analysis['prediction']}")
        print(f"   - Confidence: {analysis['confidence']:.2f}")
        print(f"   - Actual Outcome: {analysis.get('actual_outcome', 'N/A')}")
        print(f"   - Distance: {analysis['distance']:.3f}")

    # ===== STEP 6: Collection Statistics =====
    print("\n6. Collection statistics:")
    stats = store.get_collection_stats()
    print(f"   - News articles: {stats['news']}")
    print(f"   - Analyses: {stats['analyses']}")
    print(f"   - Filings: {stats['filings']}")

    # ===== STEP 7: Demonstrate Collection Reset =====
    print("\n7. Collection management:")
    print("   Resetting news collection...")
    success = store.reset_collection("news")
    if success:
        stats = store.get_collection_stats()
        print(f"   [OK] News collection reset (count: {stats['news']})")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("- VectorStore enables semantic search (not just keyword matching)")
    print("- Useful for finding related news, similar past analyses")
    print("- Can filter by ticker or search across all stocks")
    print("- Distance threshold controls match quality")
    print("- Persistent storage survives restarts")
    print("\nNext Steps:")
    print("- Integrate with SentimentAgent for context augmentation")
    print("- Use in ExplanationAgent for historical examples")
    print("- Add to workflow for enhanced analysis quality")


if __name__ == "__main__":
    main()
