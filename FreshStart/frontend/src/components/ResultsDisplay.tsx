'use client'

interface ResultsDisplayProps {
  result: any
}

export default function ResultsDisplay({ result }: ResultsDisplayProps) {
  const { prediction, sentiment, explanation, reflection } = result

  return (
    <div className="results">
      {/* Prediction Card */}
      <div className="card prediction-card">
        <h2>Prediction</h2>
        <div className="prediction-result">
          <span className={`direction ${prediction.direction}`}>
            {prediction.direction.toUpperCase()}
          </span>
          <span className="confidence">
            {(prediction.confidence * 100).toFixed(1)}% confidence
          </span>
        </div>
      </div>

      {/* Sentiment Card */}
      <div className="card sentiment-card">
        <h2>Sentiment Analysis</h2>
        <div className="sentiment-content">
          <div className={`sentiment-label ${sentiment.label}`}>
            {sentiment.label.toUpperCase()}
          </div>
          <div className="sentiment-score">
            Score: {sentiment.score.toFixed(2)}
          </div>
          {sentiment.top_headlines && sentiment.top_headlines.length > 0 && (
            <div className="headlines">
              <h3>Top Headlines:</h3>
              <ul>
                {sentiment.top_headlines.map((headline: string, index: number) => (
                  <li key={index}>{headline}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Explanation Card */}
      <div className="card explanation-card">
        <h2>Explanation</h2>
        <p>{explanation}</p>
      </div>

      {/* Reflection Card (optional - show if issues) */}
      {reflection && reflection.issues && reflection.issues.length > 0 && (
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
