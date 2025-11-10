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
    setLoading(true)
    setError(null)
    setAnalysisResult(null)

    try {
      const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker }),
      })

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status}`)
      }

      const data = await response.json()
      setAnalysisResult(data)
    } catch (err) {
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
