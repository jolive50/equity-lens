'use client'

import { useState } from 'react'
import AnalysisForm from '@/components/AnalysisForm'
import ResultsDisplay from '@/components/ResultsDisplay'

const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '')

export default function Home() {
  const [analysisResult, setAnalysisResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleAnalyze = async (ticker: string) => {
    console.log('═══════════════════════════════════════════════════════════')
    console.log('🚀 FRONTEND: Analysis Started')
    console.log('═══════════════════════════════════════════════════════════')
    console.log('📊 Ticker:', ticker)
    console.log('⏰ Timestamp:', new Date().toISOString())
    console.log('🌐 API Endpoint:', `${API_BASE_URL}/analyze`)

    setLoading(true)
    setError(null)
    setAnalysisResult(null)

    const startTime = performance.now()

    try {
      console.log('\n📤 FRONTEND: Sending request to backend...')
      console.log('Request payload:', JSON.stringify({ ticker }, null, 2))

      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker }),
      })

      const fetchTime = performance.now() - startTime
      console.log(`\n⏱️  FRONTEND: Request completed in ${fetchTime.toFixed(2)}ms`)
      console.log('📥 Response status:', response.status, response.statusText)

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status}`)
      }

      const data = await response.json()
      const totalTime = performance.now() - startTime

      console.log('\n✅ FRONTEND: Analysis completed successfully')
      console.log('⏱️  Total time:', `${totalTime.toFixed(2)}ms (${(totalTime / 1000).toFixed(2)}s)`)
      console.log('\n📊 RESULTS SUMMARY:')
      console.log('  Ticker:', data.ticker)
      console.log('  Prediction:', data.prediction?.direction, `(${(data.prediction?.confidence * 100).toFixed(1)}% confidence)`)
      console.log('  Sentiment:', data.sentiment?.label, `(score: ${data.sentiment?.score?.toFixed(2)})`)
      console.log('  Confidence Level:', data.confidence_level)
      console.log('  Warnings:', data.warnings?.length || 0)
      console.log('\n📈 PREDICTION DETAILS:')
      console.log('  Model:', data.prediction?.metadata?.model || 'Unknown')
      console.log('  Probabilities:', data.prediction?.probabilities)
      console.log('\n💬 SENTIMENT DETAILS:')
      console.log('  Trend:', data.sentiment?.trend)
      console.log('  Top Headlines:', data.sentiment?.top_headlines?.length || 0)
      console.log('\n═══════════════════════════════════════════════════════════')

      setAnalysisResult(data)
    } catch (err) {
      const errorTime = performance.now() - startTime
      console.error('\n❌ FRONTEND: Analysis failed')
      console.error('⏱️  Failed after:', `${errorTime.toFixed(2)}ms`)
      console.error('Error:', err)
      console.log('═══════════════════════════════════════════════════════════\n')
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="container">
      <header>
        <h1>FreshStart Stock Analysis</h1>
        <p>AI-powered stock predictions with sentiment analysis</p>
      </header>

      <AnalysisForm onAnalyze={handleAnalyze} loading={loading} />

      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}

      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          <p>Analyzing... This may take 10-15 seconds</p>
        </div>
      )}

      {analysisResult && <ResultsDisplay result={analysisResult} />}
    </main>
  )
}
