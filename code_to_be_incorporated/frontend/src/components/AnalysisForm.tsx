'use client'

import { useState, FormEvent } from 'react'

interface AnalysisFormProps {
  onAnalyze: (ticker: string) => void
  loading: boolean
}

export default function AnalysisForm({ onAnalyze, loading }: AnalysisFormProps) {
  const [ticker, setTicker] = useState('')

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (ticker.trim()) {
      onAnalyze(ticker.trim().toUpperCase())
    }
  }

  return (
    <form onSubmit={handleSubmit} className="analysis-form">
      <input
        type="text"
        value={ticker}
        onChange={(e) => setTicker(e.target.value)}
        placeholder="Enter ticker (e.g., AAPL)"
        maxLength={10}
        disabled={loading}
        className="ticker-input"
      />
      <button type="submit" disabled={loading || !ticker.trim()} className="analyze-btn">
        Analyze
      </button>
    </form>
  )
}
