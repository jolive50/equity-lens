import pytest
from typing import Dict

from models.sentiment.base_sentiment import (
    BaseSentiment,
    SentimentResult,
    DummySentiment,
    SentimentLabel5,
)
from models.sentiment.ensemble import SentimentEnsemble


class StaticSentiment(BaseSentiment):
    """Simple deterministic sentiment for ensemble testing."""

    def __init__(self, provider: str, probs: Dict[str, float]):
        super().__init__(provider=provider, headline_first=1.0, min_body_words=0, neutral_cap=None, neg_gate=None)
        self._probs = dict(probs)

    def _predict_text(self, text: str) -> Dict[str, float]:
        return dict(self._probs)


def test_sentiment_result_creation():
    """SentimentResult carries label, probabilities, and metadata."""
    result = SentimentResult(
        label=SentimentLabel5.POSITIVE,
        probs={"positive": 0.85, "negative": 0.05, "neutral": 0.10},
        score=0.80,
        confidence=0.85,
        provider="finbert",
        meta={"model": "TestModel"},
    )
    assert result.label == SentimentLabel5.POSITIVE
    assert result.confidence == 0.85
    assert result.probs["positive"] == 0.85
    assert result.meta["model"] == "TestModel"


def test_dummy_sentiment_handles_basic_texts():
    """DummySentiment provides quick smoke-test coverage."""
    model = DummySentiment()
    positive = model.predict_one(title="stock surges on record profit", body="")
    negative = model.predict_one(title="shares plunge after lawsuit", body="")
    neutral = model.predict_one(title="", body="")

    assert positive.probs["positive"] > positive.probs["negative"]
    assert negative.probs["negative"] > negative.probs["positive"]
    assert neutral.probs["neutral"] == pytest.approx(1.0)


def test_ensemble_combines_transformer_members():
    """Weighted ensemble should reflect provided member weights."""
    finbert = StaticSentiment("finbert", {"negative": 0.2, "neutral": 0.3, "positive": 0.5})
    deberta = StaticSentiment("deberta", {"negative": 0.5, "neutral": 0.2, "positive": 0.3})
    roberta = StaticSentiment("roberta", {"negative": 0.3, "neutral": 0.3, "positive": 0.4})

    ens = SentimentEnsemble(
        finbert=finbert,
        deberta=deberta,
        roberta=roberta,
        weights={"finbert": 0.5, "deberta": 0.3, "roberta": 0.2},
        av_trust_mode="ignore",
        entropy_weighting=False,
    )

    item = {"title": "neutral body", "body": "", "provider": "yf", "id": "sample", "word_count": 0}
    result = ens.predict_one(item)

    assert result.probs["positive"] > result.probs["negative"]
    assert result.label == 1  # positive
    info = ens.get_model_info()
    assert set(info["models"]) == {"finbert", "deberta", "roberta"}
