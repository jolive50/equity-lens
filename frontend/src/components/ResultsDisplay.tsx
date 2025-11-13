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
  const { prediction, sentiment, explanation, reflection } = result || {}

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
