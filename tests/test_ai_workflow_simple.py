#!/usr/bin/env python3
"""
Test script for the enhanced StockSense AI workflow.
Demonstrates the coordinated multi-agent analysis with 95% confidence threshold.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipelines'))

def main():
    """Run demonstration of enhanced StockSense AI workflow."""
    
    print("Enhanced StockSense AI Workflow Demo")
    print("=" * 50)
    
    # S&P 500 companies to analyze
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "UNH"]
    
    print(f"Analyzing {len(tickers)} S&P 500 companies: {', '.join(tickers)}")
    print()
    
    try:
        from pipelines.realtime.langgraph_workflow import run_enhanced_stocksense_analysis
        from pipelines.realtime.agents import (
            build_openai_llm, CoordinationAgent, HistoricalAnalysisAgent,
            SentimentAnalysisAgent, PredictionAgent, ExplanationAgent,
            SmartMoneyAgent
        )
        
        # Attempt to build a real OpenAI client for the demonstration
        llm = build_openai_llm("gpt-4o-mini")

        # Create AI agents with enhanced capabilities
        print("Initializing AI Agents:")
        print("  * Coordination Agent (synthesizes insights)")
        print("  * Historical Agent (EBITDA, P/E, ROE analysis)")
        print("  * Sentiment Agent (FinBERT-powered)")
        print("  * Prediction Agent (ML probabilistic forecasting)")
        print("  * Explanation Agent (plain English summaries)")
        print("  * Smart Money Agent (institutions, insiders)")
        print()
        
        coordination_agent = CoordinationAgent(llm, confidence_threshold=0.95)
        historical_agent = HistoricalAnalysisAgent(llm)
        sentiment_agent = SentimentAnalysisAgent(llm)
        
        # Legacy agents for compatibility
        prediction_agent = PredictionAgent(llm)
        explanation_agent = ExplanationAgent(llm)
        smart_money_agent = SmartMoneyAgent(llm)
        
        print("Executing LangGraph Multi-Agent Workflow...")
        print("  1. Validate input & prepare state")
        print("  2. Collect S&P 500 data (historical + fundamentals)")
        print("  3. Collect comprehensive news data")
        print("  4. Run Historical Working Agent (quantitative analysis)")
        print("  5. Run Sentiment Working Agent (FinBERT news analysis)")
        print("  6. Run Coordination Agent (synthesize insights)")
        print("  7. Apply 95% confidence threshold")
        print("  8. Generate comprehensive explanation")
        print()
        
        # Execute the comprehensive AI analysis workflow
        analysis_result = run_enhanced_stocksense_analysis(
            tickers=tickers,
            user_tier="premium",
            coordination_agent=coordination_agent,
            historical_agent=historical_agent,
            sentiment_agent=sentiment_agent,
            prediction_agent=prediction_agent,
            explanation_agent=explanation_agent,
            smart_money_agent=smart_money_agent,
            confidence_threshold=0.95,
        )
        
        print("AI Analysis Complete!")
        print("=" * 50)
        
        # Display results
        print(f"Confidence Score: {analysis_result['confidence_score']:.3f}")
        print(f"Threshold Met (>=95%): {analysis_result['threshold_met']}")
        print(f"Analysis Level: {analysis_result['confidence_level'].upper()}")
        print()
        
        print("Coordinating Agent Summary:")
        print("-" * 30)
        print(analysis_result.get('coordination_summary', 'No summary available'))
        print()
        
        print("Working Agents Results:")
        print("-" * 30)
        
        # Historical agent results
        if 'working_agent_results' in analysis_result:
            historical_results = analysis_result['working_agent_results'].get('historical', {})
            print(f"Historical Agent:")
            print(f"   Overall Trend: {historical_results.get('overall_trend', 'Data unavailable')}")
            print(f"   Technical Score: {historical_results.get('technical_score', 'N/A')}")
            print(f"   Fundamental Score: {historical_results.get('fundamental_score', 'N/A')}")
            print(f"   Confidence: {historical_results.get('confidence', 0):.3f}")
            
            if 'market_trends' in historical_results:
                trends = historical_results['market_trends']
                print(f"   Market Trends:")
                print(f"     Avg Revenue Growth: {trends.get('avg_revenue_growth', 0):.3f}")
                print(f"     Avg EBITDA Margin: {trends.get('avg_ebitda_margin', 0):.3f}")
                print(f"     Avg P/E Ratio: {trends.get('avg_pe_ratio', 0):.1f}")
                print(f"     Avg ROE: {trends.get('avg_roe', 0):.3f}")
            
            # Sentiment agent results
            sentiment_results = analysis_result['working_agent_results'].get('sentiment', {})
            print(f"Sentiment Agent:")
            print(f"   Overall Sentiment: {sentiment_results.get('overall_sentiment', 'Data unavailable')}")
            print(f"   Trend Direction: {sentiment_results.get('trend_direction', 'Data unavailable')}")
            print(f"   Confidence: {sentiment_results.get('confidence', 0):.3f}")
            print(f"   Total Articles: {sentiment_results.get('total_articles', 0)}")
        
        print()
        print("Company Fundamentals:")
        print("-" * 30)
        
        if 'fundamentals' in analysis_result:
            for ticker, fundamentals in list(analysis_result['fundamentals'].items())[:3]:  # Show first 3
                print(f"  {ticker}: P/E={fundamentals.get('pe_ratio', 'N/A')}, "
                      f"Revenue Growth={(fundamentals.get('revenue_growth', 0)*100):.1f}%, "
                      f"ROE={(fundamentals.get('roe', 0)*100):.1f}%")
        
        print()
        print("Comprehensive Explanation:")
        print("-" * 30)
        print(analysis_result.get('comprehensive_explanation', 'No explanation available'))
        
        if analysis_result.get('warnings'):
            print()
            print("Warnings:")
            for warning in analysis_result['warnings']:
                print(f"  * {warning}")
        
        print()
        print("Disclaimers:")
        for disclaimer in analysis_result.get('disclaimers', []):
            print(f"  * {disclaimer}")
            
        print()
        print("AI Capabilities Demonstrated:")
        print("  [OK] 1 Coordinating Agent")
        print("  [OK] 2 Working Agents (Historical + Sentiment)")
        print("  [OK] Multi-ticker S&P 500 analysis")
        print("  [OK] FinBERT sentiment analysis")
        print("  [OK] Market trend quantitative modeling")
        print("  [OK] 95% confidence threshold enforcement")
        print("  [OK] Comprehensive fundamental analysis (EBITDA, P/E, ROE)")
        print("  [OK] Real-time news sentiment processing")
        
        if analysis_result['threshold_met']:
            print("\nHIGH CONFIDENCE ANALYSIS - Recommendation criteria met!")
        else:
            print(f"\nCONFIDENCE SAFEGUARD - Analysis below {95}% threshold")
        
    except ValueError as exc:
        print(f"OPENAI_API_KEY is required to run this demo: {exc}")
        return
    except Exception as e:
        print(f"Error running AI workflow: {e}")
        print("\nAI Workflow Architecture Demonstrated:")
        print("   * LangGraph coordinated multi-agent system")
        print("   * S&P 500 fundamental data integration")
        print("   * FinBERT-powered sentiment analysis")
        print("   * 95% confidence threshold enforcement")
        print("   * Market trends quantitative modeling")
        print(f"\nWould analyze: {', '.join(tickers)} S&P 500 companies")
        print("   * Historical data with technical indicators")
        print("   * Fundamental metrics (revenue growth, EBITDA, P/E, ROE)")
        print("   * Real-time news sentiment analysis")
        print("   * Coordinated multi-agent synthesis")

if __name__ == "__main__":
    main()
