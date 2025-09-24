"use client";

import { motion } from "framer-motion";
import { Card, Inset, Badge, Button, TextField, Select, Flex, Box, Text, Heading } from "@radix-ui/themes";
import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

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
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
        throw new Error(`Analysis failed: ${response.statusText}`);
      }
      
      const result = await response.json();
      setAnalysis(result);
    } catch (err: any) {
      setError(err.message || 'Analysis failed');
    } finally {
      setLoading(false);
    }
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
            <p className="text-sm text-gray-500">Layperson-friendly stock insights</p>
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
            <Heading size="4" className="mb-4">Stock Analysis</Heading>
            <Flex gap="3" align="end">
              <Box flexGrow="1">
                <Text size="2" weight="medium" className="mb-2">Enter Stock Symbol</Text>
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
            {error && (
              <Box mt="3">
                <Text color="red" size="2">{error}</Text>
              </Box>
            )}
          </Box>
        </Card>

        {/* Analysis Results */}
        {analysis && (
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
                    <Heading size="5" className="mb-2">
                      {analysis.ticker} Forecast
                    </Heading>
                    <Text size="2" color="gray">
                      Last updated: {new Date(analysis.as_of).toLocaleString()}
                    </Text>
                  </div>
                  <Badge color={getDirectionColor(analysis.forecast.direction)} size="2">
                    {getDirectionIcon(analysis.forecast.direction)} {analysis.forecast.direction.toUpperCase()}
                  </Badge>
                </Flex>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                  <Box>
                    <Text size="2" weight="medium" color="gray">Confidence</Text>
                    <Text size="4" weight="bold">
                      {(analysis.forecast.confidence * 100).toFixed(0)}%
                    </Text>
                  </Box>
                  <Box>
                    <Text size="2" weight="medium" color="gray">95% Horizon</Text>
                    <Text size="4" weight="bold">
                      {analysis.forecast.horizon_95.days} days
                    </Text>
                  </Box>
                  <Box>
                    <Text size="2" weight="medium" color="gray">Drops Below 95%</Text>
                    <Text size="4" weight="bold">
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

            {/* Metrics Panel */}
            <Card>
              <Box p="6">
                <Heading size="4" className="mb-4">Key Metrics</Heading>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {Object.entries(analysis.metrics).map(([key, metric]) => (
                    <Box key={key} className="border rounded-lg p-4">
                      <Flex justify="between" align="start" className="mb-2">
                        <Text size="2" weight="medium" className="capitalize">
                          {key.replace(/_/g, ' ')}
                        </Text>
                        <Badge color={getVerdictColor(metric.verdict)} size="1">
                          {metric.verdict}
                        </Badge>
                      </Flex>
                      <Text size="3" weight="bold" className="mb-2">
                        {(metric.value * 100).toFixed(1)}%
                      </Text>
                      <Text size="1" color="gray">
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
                <Heading size="4" className="mb-4">News Sentiment</Heading>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <Flex align="center" gap="3" className="mb-3">
                      <Badge color={getSentimentColor(analysis.sentiment.current)} size="2">
                        {analysis.sentiment.current.toUpperCase()}
                      </Badge>
                      <Text size="2" color="gray">
                        Score: {(analysis.sentiment.score * 100).toFixed(0)}%
                      </Text>
                      <Text size="2" color="gray">
                        Trend: {analysis.sentiment.trend}
                      </Text>
                    </Flex>
                  </div>
                  <div>
                    <Text size="2" weight="medium" className="mb-2">Recent Headlines</Text>
                    <ul className="space-y-1">
                      {analysis.sentiment.headlines.map((headline, index) => (
                        <li key={index} className="text-sm text-gray-600">
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
                  <Heading size="4" className="mb-4">Smart Money Activity</Heading>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Box>
                      <Text size="2" weight="medium" className="mb-2">Institutions</Text>
                      <Text size="1" color="gray">
                        {analysis.smart_money.institutions.summary}
                      </Text>
                    </Box>
                    <Box>
                      <Text size="2" weight="medium" className="mb-2">Insiders</Text>
                      <Text size="1" color="gray">
                        {analysis.smart_money.insiders.summary}
                      </Text>
                    </Box>
                    <Box>
                      <Text size="2" weight="medium" className="mb-2">Congress</Text>
                      <Text size="1" color="gray">
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
                <Heading size="4" className="mb-4">Analysis Explanation</Heading>
                <Text size="2" className="leading-relaxed">
                  {analysis.explanation}
                </Text>
              </Box>
            </Card>

            {/* Disclaimers */}
            <Card>
              <Box p="6">
                <Heading size="4" className="mb-4">Important Disclaimers</Heading>
                <ul className="space-y-2">
                  {analysis.disclaimers.map((disclaimer, index) => (
                    <li key={index} className="text-sm text-gray-600 flex items-start">
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
          <div className="text-center text-sm text-gray-500">
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
