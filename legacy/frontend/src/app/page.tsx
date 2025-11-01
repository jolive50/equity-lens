"use client";

import { motion } from "framer-motion";
import { Card, Inset, Badge, Button, TextField, Select, Flex, Box, Text, Heading, Separator } from "@radix-ui/themes";
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
  const [userTier, setUserTier] = useState<"basic" | "registered" | "premium">("basic");
  const [watchlist, setWatchlist] = useState<string[]>([]);
  const [batch, setBatch] = useState<{ user_tier: string; count: number; items: Array<{ ticker: string; as_of: string; direction: string; confidence: number; horizon_days: number; score: number; sentiment: { current: string; score: number; trend: string; headlines: string[] } | null }> } | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [history, setHistory] = useState<Array<{ ticker: string; as_of: string; direction: string; confidence: number; score: number }>>([]);
  const [alerts, setAlerts] = useState<Array<{ ticker: string; condition: string; threshold: number }>>([]);
  const [alertTicker, setAlertTicker] = useState<string>("");
  const [alertCondition, setAlertCondition] = useState<"prob_down_gte" | "prob_up_gte" | "confidence_gte">("prob_down_gte");
  const [alertThreshold, setAlertThreshold] = useState<string>("0.7");

  // Fetch watchlist, history, alerts when tier permits
  useEffect(() => {
    const fetchScoped = async () => {
      try {
        // Watchlist for registered/premium
        if (userTier === 'registered' || userTier === 'premium') {
          const wl = await fetch('/api/watchlist').then(r => r.json()).catch(() => ({ tickers: [] }));
          setWatchlist(wl?.tickers || []);
        } else {
          setWatchlist([]);
        }
        // History for registered/premium
        if (userTier === 'registered' || userTier === 'premium') {
          const hist = await fetch('/api/history?limit=20').then(r => r.json()).catch(() => ({ items: [] }));
          setHistory(hist?.items || []);
        } else {
          setHistory([]);
        }
        // Alerts only for premium
        if (userTier === 'premium') {
          const al = await fetch('/api/alerts').then(r => r.json()).catch(() => ({ rules: [] }));
          setAlerts(al?.rules || []);
        } else {
          setAlerts([]);
        }
      } catch {}
    };
    fetchScoped();
  }, [userTier]);

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
      // refresh history after successful analyze if allowed
      if (userTier === 'registered' || userTier === 'premium') {
        try {
          const hist = await fetch('/api/history?limit=20').then(r => r.json()).catch(() => ({ items: [] }));
          setHistory(hist?.items || []);
        } catch {}
      }
    } catch (err: unknown) {
      let message = 'Analysis failed';
      if (err instanceof Error) message = err.message;
      setError(message);
      setRetryCount(prev => prev + 1);
    } finally {
      setLoading(false);
    }
  };

  const analyzeBatch = async () => {
    if (!(userTier === "registered" || userTier === "premium")) {
      setError("Batch analysis is available to registered and premium users");
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
        body: JSON.stringify({ tickers, user_tier: userTier })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.detail || 'Batch failed');
      setBatch(data);
    } catch (err: unknown) {
      let message = 'Batch failed';
      if (err instanceof Error) message = err.message;
      setError(message);
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

  const addToWatchlist = async (val: string) => {
    if (!(userTier === 'registered' || userTier === 'premium')) return;
    if (!val || watchlist.includes(val)) return;
    const res = await fetch(`/api/watchlist`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ticker: val }) });
    const data = await res.json().catch(() => ({}));
    if (res.ok) setWatchlist(data?.tickers || []);
  };

  const removeFromWatchlist = async (val: string) => {
    if (!(userTier === 'registered' || userTier === 'premium')) return;
    const res = await fetch(`/api/watchlist?ticker=${encodeURIComponent(val)}`, { method: 'DELETE' });
    const data = await res.json().catch(() => ({}));
    if (res.ok) setWatchlist(data?.tickers || []);
  };

  const createAlert = async (tickerValue: string, condition: string, threshold: number) => {
    if (userTier !== 'premium') return;
    const url = `/api/alerts`;
    const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ticker: tickerValue, condition, threshold }) });
    const data = await res.json().catch(() => ({}));
    if (res.ok) setAlerts(data?.rules || []);
  };

  const deleteAlert = async (tickerValue: string, condition?: string) => {
    if (userTier !== 'premium') return;
    const url = `/api/alerts?ticker=${encodeURIComponent(tickerValue)}${condition ? `&condition=${encodeURIComponent(condition)}` : ''}`;
    const res = await fetch(url, { method: 'DELETE' });
    const data = await res.json().catch(() => ({}));
    if (res.ok) setAlerts(data?.rules || []);
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
      <header className="bg-white border-b border-gray-200 px-6 py-4" role="banner">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">StockSense</h1>
            <p className="text-sm text-gray-700">Layperson-friendly stock insights</p>
          </div>
          <div className="flex items-center gap-4">
            <label htmlFor="userTier" className="sr-only">User Tier</label>
            <Select.Root value={userTier} onValueChange={(value: "basic" | "registered" | "premium") => setUserTier(value)}>
              <Select.Trigger id="userTier" aria-label="Select user tier" placeholder="Select tier" />
              <Select.Content>
                <Select.Item value="basic">Basic</Select.Item>
                <Select.Item value="registered">Registered</Select.Item>
                <Select.Item value="premium">Premium</Select.Item>
              </Select.Content>
            </Select.Root>
            <Badge color={userTier === "premium" ? "blue" : userTier === "registered" ? "teal" : "gray"} aria-live="polite">
              {userTier === "premium" ? "Premium" : userTier === "registered" ? "Registered" : "Basic"}
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
                <Text asChild size="2" weight="medium" className="mb-2 text-gray-800">
                  <label htmlFor="stockSymbol">Enter Stock Symbol</label>
                </Text>
                <TextField.Root
                  id="stockSymbol"
                  aria-label="Stock Symbol"
                  aria-required="true"
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
                aria-disabled={loading || !ticker.trim()}
                aria-label="Analyze stock"
              >
                {loading ? "Analyzing..." : "Analyze"}
              </Button>
            </Flex>
            {/* Watchlist Controls (Registered & Premium) */}
            <Box className="mt-6">
              <Heading size="3" className="mb-3 text-gray-900">Watchlist {(userTier === 'basic') && <Badge color="gray" className="ml-2">Registered+</Badge>}</Heading>
              <Flex gap="3" align="center" className="mb-3">
                <Text asChild className="sr-only"><label htmlFor="addWatchTicker">Add ticker</label></Text>
                <TextField.Root
                  id="addWatchTicker"
                  placeholder="Add ticker"
                  size="2"
                  onKeyDown={async (e: React.KeyboardEvent<HTMLInputElement>) => {
                    if (e.key === 'Enter') {
                      const val = String(e.currentTarget.value || '').toUpperCase().replace(/[^A-Z0-9\.\-]/g, '');
                      if (val) {
                        await addToWatchlist(val);
                        e.currentTarget.value = '';
                      }
                    }
                  }}
                  aria-label="Add ticker to watchlist"
                  disabled={!(userTier === 'registered' || userTier === 'premium')}
                  aria-disabled={!(userTier === 'registered' || userTier === 'premium')}
                />
                <Button onClick={async () => {
                  const input = (document.getElementById('addWatchTicker') as HTMLInputElement | null);
                  if (!input) return;
                  const val = String(input.value || '').toUpperCase().replace(/[^A-Z0-9\.\-]/g, '');
                  if (val) {
                    await addToWatchlist(val);
                    input.value = '';
                  }
                }} size="2" aria-label="Add ticker" disabled={!(userTier === 'registered' || userTier === 'premium')} aria-disabled={!(userTier === 'registered' || userTier === 'premium')}>Add</Button>
                <Button onClick={analyzeBatch} disabled={!(userTier === 'registered' || userTier === 'premium') || loading} size="2" color="blue" aria-disabled={!(userTier === 'registered' || userTier === 'premium') || loading} aria-label="Batch Predict">Batch Predict</Button>
                <Button onClick={downloadCsv} disabled={userTier !== 'premium'} size="2" color="green" aria-disabled={userTier !== 'premium'} aria-label="Export CSV">Export CSV</Button>
              </Flex>
              <Flex gap="2" wrap="wrap" role="list" aria-label="Watchlist tickers">
                {watchlist.map((t) => (
                  <Badge key={t} color="gray" role="listitem">
                    <span className="mr-2">{t}</span>
                    {(userTier === 'registered' || userTier === 'premium') && (
                      <button className="underline" onClick={() => removeFromWatchlist(t)} aria-label={`Remove ${t} from watchlist`}>Remove</button>
                    )}
                  </Badge>
                ))}
              </Flex>
            </Box>
            {error && (
              <Box mt="3" aria-live="assertive" role="alert">
                <ErrorMessage 
                  message={error} 
                  onDismiss={dismissError}
                />
              </Box>
            )}
          </Box>
        </Card>

        {/* History & Alerts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <Card>
            <Box p="6">
              <Heading size="4" className="mb-4 text-gray-900">Recent Searches {(userTier === 'basic') && <Badge color="gray" className="ml-2">Registered+</Badge>}</Heading>
              {(userTier === 'registered' || userTier === 'premium') ? (
                history.length === 0 ? (
                  <Text size="2" className="text-gray-700">No recent searches.</Text>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="min-w-full text-sm" role="table" aria-label="Recent searches">
                      <thead>
                        <tr className="text-left text-gray-700">
                          <th className="px-2 py-1" scope="col">Ticker</th>
                          <th className="px-2 py-1" scope="col">As Of</th>
                          <th className="px-2 py-1" scope="col">Direction</th>
                          <th className="px-2 py-1" scope="col">Confidence</th>
                          <th className="px-2 py-1" scope="col">Sentiment</th>
                        </tr>
                      </thead>
                      <tbody>
                        {history.map((h, idx) => (
                          <tr key={idx} className="border-t border-gray-200">
                            <td className="px-2 py-2 font-semibold">{h.ticker}</td>
                            <td className="px-2 py-2">{new Date(h.as_of).toLocaleString()}</td>
                            <td className="px-2 py-2">{h.direction?.toUpperCase?.() || ''}</td>
                            <td className="px-2 py-2">{(Number(h.confidence) * 100).toFixed(0)}%</td>
                            <td className="px-2 py-2">{(Number(h.score) * 100).toFixed(0)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )
              ) : (
                <Text size="2" className="text-gray-700">Sign in as Registered or Premium to view history.</Text>
              )}
            </Box>
          </Card>
          <Card>
            <Box p="6">
              <Heading size="4" className="mb-4 text-gray-900">Alerts {(userTier !== 'premium') && <Badge color="gray" className="ml-2">Premium</Badge>}</Heading>
              <Box className="mb-3">
                {userTier === 'premium' ? (
                  <Flex gap="2" align="center" wrap="wrap">
                    <TextField.Root id="alertTicker" placeholder="Ticker" size="2" aria-label="Alert ticker" value={alertTicker} onChange={(e) => setAlertTicker(e.target.value.toUpperCase().replace(/[^A-Z0-9\.\-]/g, ''))} />
                    <Select.Root value={alertCondition} onValueChange={(v: "prob_down_gte" | "prob_up_gte" | "confidence_gte") => setAlertCondition(v)}>
                      <Select.Trigger id="alertCondition" aria-label="Alert condition" />
                      <Select.Content>
                        <Select.Item value="prob_down_gte">Prob Down ≥</Select.Item>
                        <Select.Item value="prob_up_gte">Prob Up ≥</Select.Item>
                        <Select.Item value="confidence_gte">Confidence ≥</Select.Item>
                      </Select.Content>
                    </Select.Root>
                    <TextField.Root id="alertThreshold" placeholder="0.7" size="2" aria-label="Alert threshold" value={alertThreshold} onChange={(e) => setAlertThreshold(e.target.value)} />
                    <Button size="2" onClick={async () => {
                      const thr = Math.max(0, Math.min(1, Number(alertThreshold)));
                      if (!alertTicker) return;
                      await createAlert(alertTicker, alertCondition, thr);
                      setAlertTicker("");
                      setAlertThreshold("0.7");
                    }}>Add Alert</Button>
                  </Flex>
                ) : (
                  <Text size="2" className="text-gray-700">Upgrade to Premium to add alerts.</Text>
                )}
              </Box>
              <Separator size="4" className="my-3" />
              {alerts.length === 0 ? (
                <Text size="2" className="text-gray-700">No alerts configured.</Text>
              ) : (
                <ul className="space-y-2">
                  {alerts.map((r, idx) => (
                    <li key={idx} className="text-sm text-gray-700 flex items-center justify-between">
                      <span>{r.ticker} — {r.condition} — {(r.threshold * 100).toFixed(0)}%</span>
                      {userTier === 'premium' && (
                        <button className="underline" onClick={() => deleteAlert(r.ticker, r.condition)} aria-label={`Delete alert for ${r.ticker}`}>Delete</button>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </Box>
          </Card>
        </div>

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

        {/* Batch Results (Registered & Premium) */}
        {batch && (userTier === 'registered' || userTier === 'premium') && (
          <Card className="mb-6">
            <Box p="6">
              <Heading size="4" className="mb-4 text-gray-900">Watchlist Predictions</Heading>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm" role="table" aria-label="Watchlist predictions table">
                  <thead>
                    <tr className="text-left text-gray-700">
                      <th className="px-2 py-1" scope="col">Ticker</th>
                      <th className="px-2 py-1" scope="col">As Of</th>
                      <th className="px-2 py-1" scope="col">Direction</th>
                      <th className="px-2 py-1" scope="col">Confidence</th>
                      <th className="px-2 py-1" scope="col">95% Horizon (days)</th>
                      <th className="px-2 py-1" scope="col">Sentiment Score</th>
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
                  <Badge color={getDirectionColor(analysis.forecast.direction)} size="2" aria-label={`Direction ${analysis.forecast.direction}`}>
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
