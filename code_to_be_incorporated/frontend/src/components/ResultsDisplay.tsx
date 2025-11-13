'use client'

interface Prediction {
  direction?: string
  confidence?: number
  [key: string]: any
}

interface Sentiment {
  label?: string
  score?: number
  top_headlines?: string[]
  [key: string]: any
}

interface Reflection {
  issues?: string[]
  [key: string]: any
}

interface AnalysisResult {
  ticker?: string
  prediction?: Prediction
  sentiment?: Sentiment
  final_decision?: string
  explanation?: string
  reflection?: Reflection | null
}

interface ResultsDisplayProps {
  result: AnalysisResult
}

export default function ResultsDisplay({ result }: ResultsDisplayProps) {
  const { prediction, sentiment, explanation, reflection, ticker } = result || {}

  // Demo price data for sparkline chart (frontend-only dummy data)
  const demoPrices = [225, 227, 223, 230, 232, 228, 235, 238]

  // SVG용 좌표 변환
  // Convert prices to SVG polyline points
  const createSparklinePoints = (values: number[]): string => {
    if (!values || values.length === 0) return ''

    const height = 40
    const padding = 6
    const widthStep = 100 / (values.length - 1)
    const max = Math.max(...values)
    const min = Math.min(...values)
    const range = max - min || 1 // 0으로 나누기 방지 / avoid division by zero

    return values
      .map((value, index) => {
        const x = widthStep * index

        // Larger value -> smaller y (higher on chart)
        const normalized = (max - value) / range
        const y = padding + normalized * (height - padding * 2)
        return `${x.toFixed(1)},${y.toFixed(1)}`
      })
      .join(' ')
  }

  const sparklinePoints = createSparklinePoints(demoPrices)

  const hasPrediction = !!prediction && typeof prediction.direction === 'string'
  const hasSentiment = !!sentiment && typeof sentiment.label === 'string'

  return (
    <div className="results">
      {/* 🔹 Prediction Card / 예측 카드 */}
      <div className="card prediction-card">
        <h2>Prediction</h2>
        {hasPrediction ? (
          <>
            <div className="prediction-result">
              <span
                className={`direction ${
                  prediction!.direction === 'up'
                    ? 'up'
                    : prediction!.direction === 'down'
                    ? 'down'
                    : 'neutral'
                }`}
              >
                {/* Example: UP / DOWN / STEADY */}
                {prediction!.direction?.toUpperCase()}
              </span>
              {typeof prediction!.confidence === 'number' && (
                <span className="confidence">
                  {(prediction!.confidence * 100).toFixed(1)}% confidence
                </span>
              )}
            </div>

            {/* Demo mini chart using dummy AAPL-style prices */}
            <div
              style={{
                marginTop: '0.8rem',
                height: '40px',
                background:
                  'linear-gradient(135deg, rgba(56,189,248,0.16), rgba(15,23,42,0.4))',
                borderRadius: '0.6rem',
                position: 'relative',
                overflow: 'hidden',
                border: '1px solid rgba(148,163,184,0.3)',
              }}
            >
              <svg
                width="100%"
                height="40"
                viewBox="0 0 100 40"
                preserveAspectRatio="none"
              >
                <polyline
                  points={sparklinePoints}
                  fill="none"
                  stroke="#38bdf8"
                  strokeWidth={2.2}
                  strokeLinecap="round"
                />
              </svg>
              <div
                style={{
                  position: 'absolute',
                  top: 4,
                  left: 8,
                  fontSize: '0.7rem',
                  color: 'var(--muted)',
                }}
              >
                {/* 여기서는 그냥 데모용 텍스트 / Just demo label */}
                {ticker ? `${ticker} demo trend` : 'Demo price trend'}
              </div>
            </div>
          </>
        ) : (
          <p>No prediction data.</p>
        )}
      </div>

      {/* 🔹 Sentiment Card / 감성 분석 카드 */}
      <div className="card sentiment-card">
        <h2>Sentiment Analysis</h2>

        {hasSentiment ? (
          <div className="sentiment-content">
            <div className={`sentiment-label ${sentiment!.label}`}>
              {sentiment!.label?.toUpperCase()}
            </div>
            {typeof sentiment!.score === 'number' && (
              <div className="sentiment-score">
                Score: {sentiment!.score.toFixed(2)}
              </div>
            )}

            {/* This part is optional – only render if backend sends headlines */}
            {Array.isArray(sentiment!.top_headlines) &&
              sentiment!.top_headlines.length > 0 && (
                <div className="headlines">
                  <h3>Top Headlines:</h3>
                  <ul>
                    {sentiment!.top_headlines.map(
                      (headline: string, index: number) => (
                        <li key={index}>{headline}</li>
                      ),
                    )}
                  </ul>
                </div>
              )}
          </div>
        ) : (
          <p>No sentiment data.</p>
        )}
      </div>

      {/* 🔹 Explanation Card / 설명 카드 */}
      <div className="card explanation-card">
        <h2>Explanation</h2>
        <p>{explanation || 'No explanation available.'}</p>
      </div>

      {/* 🔹 Reflection Card (optional) / 리플렉션 카드 (있을 때만) */}
      {reflection && Array.isArray(reflection.issues) && reflection.issues.length > 0 && (
        <div className="card reflection-card">
          <h2>Quality Notes</h2>
          <ul>
            {reflection.issues.map((issue: string, index: number) => (
              <li key={index}>{issue}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
