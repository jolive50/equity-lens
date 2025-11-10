"""FastAPI backend for FreshStart stock analysis.

SUA's Component - Backend API
Integrates with LangGraph workflow to provide stock analysis endpoint.
"""
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from api.models import (
    AnalysisRequest,
    AnalysisResponse,
    ErrorResponse,
    PredictionResult,
    SentimentResult,
    ReflectionResult
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FreshStart Stock Analysis API",
    description="AI-powered stock predictions with sentiment analysis",
    version="1.0.0"
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "service": "FreshStart API",
        "status": "running",
        "version": "1.0.0"
    }


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    }
)
async def analyze_stock(request: AnalysisRequest) -> AnalysisResponse:
    """Analyze a stock using the FreshStart workflow.

    Args:
        request: Analysis request with ticker and user_tier

    Returns:
        Complete analysis results including prediction, sentiment, and explanation

    Raises:
        HTTPException: If analysis fails or invalid input
    """
    ticker = request.ticker
    user_tier = request.user_tier

    logger.info(f"Received analysis request: ticker={ticker}, tier={user_tier}")

    try:
        # Import workflow (lazy import to avoid startup delays)
        from coordinator.workflow import run_stock_analysis

        # Run the analysis workflow
        result = run_stock_analysis(
            ticker=ticker,
            user_tier=user_tier
        )

        # Transform workflow result to API response format
        response = _transform_workflow_result(result)

        logger.info(f"Analysis completed for {ticker}: {response.prediction.direction}")
        return response

    except ValueError as e:
        # Validation errors from workflow
        logger.error(f"Validation error for {ticker}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "ValidationError",
                "message": str(e),
                "details": {"ticker": ticker}
            }
        )

    except Exception as e:
        # Unexpected errors
        logger.error(f"Analysis failed for {ticker}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "AnalysisError",
                "message": f"Failed to analyze {ticker}: {str(e)}",
                "details": {"ticker": ticker}
            }
        )


def _transform_workflow_result(result: Dict[str, Any]) -> AnalysisResponse:
    """Transform workflow result to API response format.

    Args:
        result: Raw workflow output

    Returns:
        Structured AnalysisResponse
    """
    # Extract prediction data
    prediction_data = result.get("prediction", {})
    prediction = PredictionResult(
        direction=prediction_data.get("direction", "neutral"),
        confidence=prediction_data.get("confidence", 0.5),
        narrative=prediction_data.get("narrative", ""),
        probabilities=prediction_data.get("probabilities", {"up": 0.33, "down": 0.33, "neutral": 0.34}),
        metadata=prediction_data.get("metadata", {})
    )

    # Extract sentiment data and transform to frontend format
    sentiment_data = result.get("sentiment", {})
    sentiment = SentimentResult(
        label=sentiment_data.get("current", "neutral"),  # Map "current" to "label"
        score=sentiment_data.get("score", 0.0),
        trend=sentiment_data.get("trend", "stable"),
        top_headlines=sentiment_data.get("headlines", [])  # Map "headlines" to "top_headlines"
    )

    # Extract reflection data
    reflection_data = result.get("metadata", {})
    warnings = result.get("warnings", [])
    reflection = ReflectionResult(
        validation_passed=reflection_data.get("reflection_passed", True),
        issues=warnings  # Use warnings as reflection issues
    )

    # Build complete response
    response = AnalysisResponse(
        ticker=result.get("ticker", ""),
        as_of=result.get("as_of", ""),
        prediction=prediction,
        sentiment=sentiment,
        explanation=result.get("explanation", ""),
        confidence_level=result.get("confidence_level", "medium"),
        reflection=reflection,
        warnings=warnings,
        metadata=result.get("metadata", {})
    )

    return response


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError) -> JSONResponse:
    """Handle Pydantic validation errors."""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "ValidationError",
            "message": "Invalid request data",
            "details": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception) -> JSONResponse:
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "details": {"error": str(exc)}
        }
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FreshStart API server...")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
