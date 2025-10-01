"use client";

import { motion } from "framer-motion";
import { Card, Inset, Badge, Button, TextField, Select, Flex, Box, Text, Heading } from "@radix-ui/themes";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { LoadingCard, LoadingSpinner } from "../components/Loading";
import { DailyProbabilitiesChart, ConfidenceHorizonCard } from "../components/Charts";
import { ErrorCard, ErrorMessage } from "../components/ErrorHandling";

type AnalysisResult = {
  ticker: string;
  as_of: string;
  forecast: {
    direction: "up" | "down" | "neutral";
    confidence: number;
    horizon_95: {
      days: number;
      end_date: string;
      drops_below_95_on: string;
    };
    daily_probs: Array<{
      date: string;
      up: number;
      down: number;
      neutral: number;
    }>;
  };
  metrics: Record<string, {
    value: number;
    verdict: "Good" | "OK" | "Needs caution";
    explanation: string;
  }>;
  sentiment: {
    current: "positive" | "neutral" | "negative";
    score: number;
    trend: "improving" | "stable" | "declining";
    headlines: string[];
  };
  smart_money: {
    institutions: { summary: string };
    insiders: { summary: string };
    congress: { summary: string };
  };
  explanation: string;
  warnings: string[];
  disclaimers: string[];
};

export default function Home() {
  const [ticker, setTicker] = useState("AAPL");
  const [userTier, setUserTier] = useState<"basic" | "premium">("basic");
  const [watchlist, setWatchlist] = useState<string[]>(["AAPL", "MSFT", "GOOGL"]);
  const [batch, setBatch] = useState<{ user_tier: string; count: number; items: Array<{ ticker: string; as_of: string; direction: string; confidence: number; horizon_days: number; score: number; sentiment: any }>} | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  const analyzeStock = async () => {
    if (!ticker.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticker: ticker.toUpperCase(),
          user_tier: userTier,
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Analysis failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      setAnalysis(result);
      setRetryCount(0); // Reset retry count on success
    } catch (err: any) {
      setError(err.message || 'Analysis failed');
      setRetryCount(prev => prev + 1);
    } finally {
      setLoading(false);
    }
  };

  const analyzeBatch = async () => {
    if (userTier !== "premium") {
      setError("Batch analysis is available to premium users only");
      return;
    }
    const tickers = watchlist.map(t => t.toUpperCase()).filter(Boolean);
    if (!tickers.length) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tickers, user_tier: 'premium' })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || 'Batch failed');
      setBatch(data);
    } catch (err: any) {
      setError(err?.message || 'Batch failed');
    } finally {
      setLoading(false);
    }
  };

  const downloadCsv = () => {
    if (userTier !== "premium") {
      setError("CSV export is available to premium users only");
      return;
    }
    const tickers = encodeURIComponent(watchlist.join(','));
    const url = `/api/export?tickers=${tickers}&max_rows=5000&user_tier=premium`;
    window.location.href = url;
  };

  const handleRetry = () => {
    setError(null);
    analyzeStock();
  };

  const dismissError = () => {
    setError(null);
  };

  const getDirectionColor = (direction: string) => {
    switch (direction) {
      case "up": return "green";
      case "down": return "red";
      default: return "gray";
    }
  };

  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case "up": return "📈";
      case "down": return "📉";
      default: return "➡️";
    }
  };

  const getVerdictColor = (verdict: string) => {
    switch (verdict) {
      case "Good": return "green";
      case "OK": return "yellow";
      case "Needs caution": return "red";
      default: return "gray";
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case "positive": return "green";
      case "negative": return "red";
      default: return "gray";
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">StockSense</h1>
            <p className="text-sm text-gray-700">Layperson-friendly stock insights</p>
          </div>
          <div className="flex items-center gap-4">
            <Select.Root value={userTier} onValueChange={(value: "basic" | "premium") => setUserTier(value)}>
              <Select.Trigger placeholder="Select tier" />
              <Select.Content>
                <Select.Item value="basic">Basic</Select.Item>
                <Select.Item value="premium">Premium</Select.Item>
              </Select.Content>
            </Select.Root>
            <Badge color={userTier === "premium" ? "blue" : "gray"}>
              {userTier === "premium" ? "Premium" : "Basic"}
            </Badge>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Search Section */}
        <Card className="mb-8">
          <Box p="6">
            <Heading size="4" className="mb-4 text-gray-900">Stock Analysis</Heading>
            <Flex gap="3" align="end">
              <Box flexGrow="1">
                <Text size="2" weight="medium" className="mb-2 text-gray-800">Enter Stock Symbol</Text>
                <TextField.Root
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  placeholder="e.g., AAPL, MSFT, TSLA"
                  size="3"
                />
              </Box>
              <Button 
                onClick={analyzeStock} 
                disabled={loading || !ticker.trim()}
                size="3"
              >
                {loading ? "Analyzing..." : "Analyze"}
              </Button>
            </Flex>
            {/* Watchlist Controls (Premium) */}
            <Box className="mt-6">
              <Heading size="3" className="mb-3 text-gray-900">Watchlist {userTier !== 'premium' && <Badge color="gray" className="ml-2">Premium</Badge>}</Heading>
              <Flex gap="3" align="center" className="mb-3">
                <TextField.Root
                  placeholder="Add ticker"
                  size="2"
                  onKeyDown={(e: any) => {
                    if (e.key === 'Enter') {
                      const val = String(e.currentTarget.value || '').toUpperCase().replace(/[^A-Z0-9\.\-]/g, '');
                      if (val && !watchlist.includes(val)) {
                        setWatchlist(prev => [...prev, val]);
                        e.currentTarget.value = '';
                      }
                    }
                  }}
                />
                <Button onClick={() => {
                  const input = (document.activeElement as HTMLInputElement);
                  if (!input || !('value' in input)) return;
                  const val = String((input as any).value || '').toUpperCase().replace(/[^A-Z0-9\.\-]/g, '');
                  if (val && !watchlist.includes(val)) {
                    setWatchlist(prev => [...prev, val]);
                    (input as any).value = '';
                  }
                }} size="2">Add</Button>
                <Button onClick={analyzeBatch} disabled={userTier !== 'premium' || loading} size="2" color="blue">Batch Predict</Button>
                <Button onClick={downloadCsv} disabled={userTier !== 'premium'} size="2" color="green">Export CSV</Button>
              </Flex>
              <Flex gap="2" wrap="wrap">
                {watchlist.map((t) => (
                  <Badge key={t} color="gray">
                    {t}
                  </Badge>
                ))}
              </Flex>
            </Box>
            {error && (
              <Box mt="3">
                <ErrorMessage 
                  message={error} 
                  onDismiss={dismissError}
                />
              </Box>
            )}
          </Box>
        </Card>

        {/* Loading State */}
        {loading && (
          <LoadingCard 
            message="Analyzing stock data..."
            submessage="This may take a few moments while we gather market data and run our AI models."
          />
        )}

        {/* Error State */}
        {error && !loading && (
          <ErrorCard
            title="Analysis Failed"
            message={error}
            onRetry={retryCount < 3 ? handleRetry : undefined}
          />
        )}

        {/* Batch Results (Premium) */}
        {batch && userTier === 'premium' && (
          <Card className="mb-6">
            <Box p="6">
              <Heading size="4" className="mb-4 text-gray-900">Watchlist Predictions</Heading>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="text-left text-gray-700">
                      <th className="px-2 py-1">Ticker</th>
                      <th className="px-2 py-1">As Of</th>
                      <th className="px-2 py-1">Direction</th>
                      <th className="px-2 py-1">Confidence</th>
                      <th className="px-2 py-1">95% Horizon (days)</th>
                      <th className="px-2 py-1">Sentiment Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {batch.items.map((it, idx) => (
                      <tr key={idx} className="border-t border-gray-200">
                        <td className="px-2 py-2 font-semibold">{it.ticker}</td>
                        <td className="px-2 py-2">{new Date(it.as_of).toLocaleString()}</td>
                        <td className="px-2 py-2">{it.direction.toUpperCase()}</td>
                        <td className="px-2 py-2">{(it.confidence * 100).toFixed(0)}%</td>
                        <td className="px-2 py-2">{it.horizon_days}</td>
                        <td className="px-2 py-2">{(it.score * 100).toFixed(0)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Box>
          </Card>
        )}

        {/* Analysis Results */}
        {analysis && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Forecast Card */}
            <Card>
              <Box p="6">
                <Flex justify="between" align="start" className="mb-4">
                  <div>
                    <Heading size="5" className="mb-2 text-gray-900">
                      {analysis.ticker} Forecast
                    </Heading>
                    <Text size="2" className="text-gray-700">
                      Last updated: {new Date(analysis.as_of).toLocaleString()}
                    </Text>
                  </div>
                  <Badge color={getDirectionColor(analysis.forecast.direction)} size="2">
                    {getDirectionIcon(analysis.forecast.direction)} {analysis.forecast.direction.toUpperCase()}
                  </Badge>
                </Flex>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <Box>
                    <Text size="2" weight="medium" className="text-gray-800">Confidence</Text>
                    <Text size="4" weight="bold" className="text-gray-900">
                      {(analysis.forecast.confidence * 100).toFixed(0)}%
                    </Text>
                  </Box>
                  <Box>
                    <Text size="2" weight="medium" className="text-gray-800">95% Horizon</Text>
                    <Text size="4" weight="bold" className="text-gray-900">
                      {analysis.forecast.horizon_95.days} days
                    </Text>
                  </Box>
                  <Box>
                    <Text size="2" weight="medium" className="text-gray-800">Drops Below 95%</Text>
                    <Text size="4" weight="bold" className="text-gray-900">
                      {analysis.forecast.horizon_95.drops_below_95_on}
                    </Text>
                  </Box>
                </div>

                {analysis.warnings.length > 0 && (
                  <Box className="mb-4">
                    {analysis.warnings.map((warning, index) => (
                      <Badge key={index} color="yellow" className="mr-2 mb-2">
                        ⚠️ {warning}
                      </Badge>
                    ))}
                  </Box>
                )}
              </Box>
            </Card>

            {/* Charts Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <DailyProbabilitiesChart dailyProbs={analysis.forecast.daily_probs || []} />
              <ConfidenceHorizonCard horizon95={analysis.forecast.horizon_95} />
            </div>

            {/* Metrics Panel */}
            <Card>
              <Box p="6">
                <Heading size="4" className="mb-4 text-gray-900">Key Metrics</Heading>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(analysis.metrics).map(([key, metric]) => (
                    <Box key={key} className="border rounded-lg p-4">
                      <Flex justify="between" align="start" className="mb-2">
                        <Text size="2" weight="medium" className="capitalize text-gray-800">
                          {key.replace(/_/g, ' ')}
                        </Text>
                        <Badge color={getVerdictColor(metric.verdict)} size="1">
                          {metric.verdict}
                        </Badge>
                      </Flex>
                      <Text size="3" weight="bold" className="mb-2 text-gray-900">
                        {(metric.value * 100).toFixed(1)}%
                      </Text>
                      <Text size="1" className="text-gray-700">
                        {metric.explanation}
                      </Text>
                    </Box>
                  ))}
                </div>
              </Box>
            </Card>

            {/* Sentiment Panel */}
            <Card>
              <Box p="6">
                <Heading size="4" className="mb-4 text-gray-900">News Sentiment</Heading>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <Flex align="center" gap="3" className="mb-3">
                      <Badge color={getSentimentColor(analysis.sentiment.current)} size="2">
                        {analysis.sentiment.current.toUpperCase()}
                      </Badge>
                      <Text size="2" className="text-gray-700">
                        Score: {(analysis.sentiment.score * 100).toFixed(0)}%
                      </Text>
                      <Text size="2" className="text-gray-700">
                        Trend: {analysis.sentiment.trend}
                      </Text>
                    </Flex>
                  </div>
                  <div>
                    <Text size="2" weight="medium" className="mb-2 text-gray-800">Recent Headlines</Text>
                    <ul className="space-y-1">
                      {analysis.sentiment.headlines.map((headline, index) => (
                        <li key={index} className="text-sm text-gray-700">
                          • {headline}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </Box>
            </Card>

            {/* Smart Money Panel (Premium only) */}
            {userTier === "premium" && (
              <Card>
                <Box p="6">
                  <Heading size="4" className="mb-4 text-gray-900">Smart Money Activity</Heading>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Box>
                      <Text size="2" weight="medium" className="mb-2 text-gray-800">Institutions</Text>
                      <Text size="1" className="text-gray-700">
                        {analysis.smart_money.institutions.summary}
                      </Text>
                    </Box>
                    <Box>
                      <Text size="2" weight="medium" className="mb-2 text-gray-800">Insiders</Text>
                      <Text size="1" className="text-gray-700">
                        {analysis.smart_money.insiders.summary}
                      </Text>
                    </Box>
                    <Box>
                      <Text size="2" weight="medium" className="mb-2 text-gray-800">Congress</Text>
                      <Text size="1" className="text-gray-700">
                        {analysis.smart_money.congress.summary}
                      </Text>
                    </Box>
                  </div>
                </Box>
              </Card>
            )}

            {/* Explanation */}
            <Card>
              <Box p="6">
                <Heading size="4" className="mb-4 text-gray-900">Analysis Explanation</Heading>
                <Text size="2" className="leading-relaxed text-gray-800 mb-3">
                  {analysis.explanation}
                </Text>
                {userTier === 'premium' && (
                  <Box className="mt-2">
                    <Heading size="3" className="mb-2 text-gray-900">Contributing Factors</Heading>
                    <ul className="list-disc ml-5 text-gray-800">
                      {Object.entries(analysis.metrics).slice(0,3).map(([key, metric]) => (
                        <li key={key}>
                          <span className="font-medium capitalize">{key.replace(/_/g, ' ')}</span>: {metric.explanation}
                        </li>
                      ))}
                    </ul>
                  </Box>
                )}
              </Box>
            </Card>

            {/* Disclaimers */}
            <Card>
              <Box p="6">
                <Heading size="4" className="mb-4 text-gray-900">Important Disclaimers</Heading>
                <ul className="space-y-2">
                  {analysis.disclaimers.map((disclaimer, index) => (
                    <li key={index} className="text-sm text-gray-700 flex items-start">
                      <span className="text-red-500 mr-2">⚠️</span>
                      {disclaimer}
                    </li>
                  ))}
                </ul>
              </Box>
            </Card>
          </motion.div>
        )}

        {/* Footer */}
        <footer className="mt-12 pt-8 border-t border-gray-200">
          <div className="text-center text-sm text-gray-700">
            <p>StockSense - Informational purposes only. Not investment advice.</p>
            <p className="mt-2">
              Explore more on <Link className="text-indigo-400 hover:underline" href="/news">News</Link> and <Link className="text-indigo-400 hover:underline" href="/dashboard">Dashboard</Link>.
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}
