import { NextResponse } from "next/server";

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { ticker, user_tier } = body;

    if (!ticker) {
      return NextResponse.json(
        { error: "Ticker is required" },
        { status: 400 }
      );
    }

    // Mock response for demonstration
    // In production, this would call the actual StockSense API
    const mockResponse = {
      ticker: ticker.toUpperCase(),
      as_of: new Date().toISOString(),
      forecast: {
        direction: "up",
        confidence: 0.87,
        horizon_95: {
          days: 7,
          end_date: "2025-02-01",
          drops_below_95_on: "2025-02-02"
        },
        daily_probs: [
          {
            date: "2025-01-25",
            up: 0.71,
            down: 0.18,
            neutral: 0.11
          },
          {
            date: "2025-01-26",
            up: 0.68,
            down: 0.20,
            neutral: 0.12
          }
        ]
      },
      metrics: {
        revenue_growth: {
          value: 0.08,
          verdict: "Good",
          explanation: "Revenue growth is above sector average and 5-year median."
        },
        ebitda_margin: {
          value: 0.28,
          verdict: "Good",
          explanation: "EBITDA margin is strong and improving over recent quarters."
        },
        debt_to_ebitda: {
          value: 2.1,
          verdict: "OK",
          explanation: "Debt levels are moderate - watch interest coverage."
        }
      },
      sentiment: {
        current: "positive",
        score: 0.72,
        trend: "improving",
        headlines: [
          "Strong earnings report exceeds expectations",
          "New product launch drives optimism",
          "Analyst upgrades price target"
        ]
      },
      smart_money: {
        institutions: {
          summary: "Institutional ownership increased 2% last quarter with Vanguard and BlackRock leading purchases."
        },
        insiders: {
          summary: "Recent insider activity shows confidence with net buying of 50,000 shares."
        },
        congress: {
          summary: "No recent congressional trading activity reported."
        }
      },
      explanation: `Based on our analysis, ${ticker.toUpperCase()} shows strong upward momentum with 87% confidence over the next 7 days. The company's fundamentals are solid with revenue growth above sector average and strong EBITDA margins. News sentiment is positive, driven by strong earnings and new product launches. Institutional investors have been net buyers, showing confidence in the company's prospects. However, debt levels are moderate and should be monitored. This analysis is for informational purposes only and not investment advice.`,
      warnings: [],
      disclaimers: [
        "This is informational only, not investment advice",
        "Data sources: Alpha Vantage, NewsAPI, SEC filings",
        `Last updated: ${new Date().toISOString()}`
      ]
    };

    return NextResponse.json(mockResponse);
  } catch (error: any) {
    return NextResponse.json(
      { error: "Analysis failed" },
      { status: 500 }
    );
  }
}
