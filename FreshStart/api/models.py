"""Pydantic models for FreshStart API.

SUA's Component - API Request/Response Models
Type-safe schemas for FastAPI endpoints.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class AnalysisRequest(BaseModel):
    """Request model for stock analysis."""

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Stock ticker symbol (e.g., AAPL, TSLA)"
    )
    user_tier: str = Field(
        default="basic",
        description="User subscription tier: basic or premium"
    )

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: str) -> str:
        """Validate and normalize ticker."""
        if not v or not v.strip():
            raise ValueError("Ticker cannot be empty")

        ticker = v.strip().upper()

        if not ticker.isalnum():
            raise ValueError("Ticker must contain only letters and numbers")

        return ticker

    @field_validator("user_tier")
    @classmethod
    def validate_user_tier(cls, v: str) -> str:
        """Validate user tier."""
        valid_tiers = ["basic", "premium"]
        if v not in valid_tiers:
            raise ValueError(f"User tier must be one of: {valid_tiers}")
        return v


class PredictionResult(BaseModel):
    """Prediction analysis result."""

    direction: str = Field(
        description="Predicted direction: up, down, or neutral"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )
    narrative: str = Field(
        description="Brief narrative explanation"
    )
    probabilities: Dict[str, float] = Field(
        description="Probability distribution across directions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (model info, etc.)"
    )


class SentimentResult(BaseModel):
    """Sentiment analysis result."""

    label: str = Field(
        description="Sentiment label: positive, negative, or neutral"
    )
    score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Sentiment score between -1 (negative) and 1 (positive)"
    )
    trend: str = Field(
        description="Sentiment trend: improving, declining, or stable"
    )
    top_headlines: List[str] = Field(
        default_factory=list,
        description="Top relevant news headlines"
    )


class ReflectionResult(BaseModel):
    """Reflection/validation result."""

    validation_passed: bool = Field(
        description="Whether validation checks passed"
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of validation issues or warnings"
    )


class AnalysisResponse(BaseModel):
    """Complete analysis response."""

    ticker: str = Field(
        description="Stock ticker analyzed"
    )
    as_of: str = Field(
        description="Timestamp of analysis (ISO 8601 format)"
    )
    prediction: PredictionResult = Field(
        description="Prediction analysis results"
    )
    sentiment: SentimentResult = Field(
        description="Sentiment analysis results"
    )
    explanation: str = Field(
        description="Natural language explanation of the analysis"
    )
    confidence_level: str = Field(
        description="Overall confidence level: low, medium, or high"
    )
    reflection: ReflectionResult = Field(
        description="Quality validation results"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Any warnings or issues encountered"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(
        description="Error type"
    )
    message: str = Field(
        description="Human-readable error message"
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
