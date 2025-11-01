# benchmark_finnews.py
# Benchmark your sentiment pipeline on human-labeled datasets (Financial PhraseBank + FiQA)
# and optionally evaluate AV/ensemble vs human labels in your Alpha Vantage CSV.
#
# Notes:
# - Reproducible shuffling BEFORE truncation to avoid biased slices.
# - Presets and CLI overrides apply to your imported pipeline at runtime.
# - The pipeline’s vote_probs now respects the passed neutral_cap (as fixed in the pipeline).

import argparse
import csv
import importlib.util
import json
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

# -----------------------------
# Dynamic import of your pipeline
# -----------------------------
def load_pipeline_module(pipeline_path: str):
    p = pathlib.Path(pipeline_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Pipeline script not found: {p}")
    spec = importlib.util.spec_from_file_location("sent_pipeline", str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Could not prepare module loader"
    spec.loader.exec_module(mod)
    # sanity check: required symbols
    required = [
        "vote_probs",
        "finbert_probs",
        "roberta_probs",
        "vader_probs",
        "textblob_probs",
        "WEIGHTS_BASE",
        "NEUTRAL_CAP",
        "FB_TEMP",
        "FB_HEAD_W",
    ]
    for name in required:
        if not hasattr(mod, name):
            raise RuntimeError(f"Pipeline script missing required symbol: {name}")
    # Optional knobs with sensible defaults
    if not hasattr(mod, "RB_HEAD_W"):
        setattr(mod, "RB_HEAD_W", 1.0)
    if not hasattr(mod, "SOFTEN_TAU"):
        setattr(mod, "SOFTEN_TAU", 1.0)
    return mod

# -----------------------------
# Apply runtime presets / overrides to the pipeline (no edits needed in pipeline file)
# -----------------------------
def apply_preset(pipe, preset: Optional[str]):
    """Domain presets adjust a few knobs on the imported pipeline in-memory."""
    if not preset:
        return
    preset = preset.lower()
    if preset == "news":
        return
    if preset == "fiqa":
        try:
            if hasattr(pipe, "NEUTRAL_CAP"):
                pipe.NEUTRAL_CAP = 0.28
            if hasattr(pipe, "SOFTEN_TAU"):
                pipe.SOFTEN_TAU = 0.98
            if hasattr(pipe, "WEIGHTS_BASE"):
                w = dict(pipe.WEIGHTS_BASE)
                # prefer RoBERTa a bit more, shrink lexicons slightly
                w.update({"roberta": 0.38, "vader": 0.03, "textblob": 0.03})
                s = sum(w.values())
                w = {k: v / s for k, v in w.items()}
                pipe.WEIGHTS_BASE = w
        except Exception:
            pass

def apply_overrides(pipe, neutral_cap: Optional[float], soften_tau: Optional[float], override_weights: Optional[str]):
    """Command-line overrides for fine control without editing the pipeline."""
    if neutral_cap is not None and hasattr(pipe, "NEUTRAL_CAP"):
        pipe.NEUTRAL_CAP = float(neutral_cap)
    if soften_tau is not None and hasattr(pipe, "SOFTEN_TAU"):
        pipe.SOFTEN_TAU = float(soften_tau)
    if override_weights:
        try:
            ow = json.loads(override_weights)
            if not isinstance(ow, dict):
                raise ValueError("override_weights must be a JSON object")
            w = dict(getattr(pipe, "WEIGHTS_BASE", {}))
            w.update({str(k): float(v) for k, v in ow.items()})
            s = sum(w.values())
            if s <= 0:
                raise ValueError("override_weights sum must be > 0")
            pipe.WEIGHTS_BASE = {k: v / s for k, v in w.items()}
        except Exception as e:
            raise RuntimeError(f"Failed to parse --override-weights: {e}") from e

# -----------------------------
# HuggingFace datasets loader
# -----------------------------
def _safe_import_datasets():
    try:
        from datasets import load_dataset  # type: ignore
        return load_dataset
    except Exception as e:
        raise RuntimeError(
            "HuggingFace 'datasets' is required. Install with `pip install datasets`."
        ) from e

# -----------------------------
# Label helpers
# -----------------------------
_POS = "positive"
_NEU = "neutral"
_NEG = "negative"
EVAL_LABELS = [_NEG, _NEU, _POS]  # evaluation expects this order

# IMPORTANT: your model prob vectors are ordered [positive, neutral, negative]
MODEL_PROB_LABELS = [_POS, _NEU, _NEG]

def coerce_label(x) -> str:
    if x is None:
        return _NEU
    s = str(x).strip().lower()
    # numeric encodings
    if s in {"2", "pos", "positive", "bullish"}:
        return _POS
    if s in {"1", "neu", "neutral"}:
        return _NEU
    if s in {"0", "neg", "negative", "bearish"}:
        return _NEG
    # continuous
    try:
        v = float(s)
        if v > 0.05:
            return _POS
        if v < -0.05:
            return _NEG
        return _NEU
    except Exception:
        if "pos" in s:
            return _POS
        if "neg" in s:
            return _NEG
        return _NEU

def pred_from_probs(prob_vec: np.ndarray) -> Tuple[str, float]:
    """
    Map argmax index from model prob vectors (ordered [pos, neu, neg]) to label string.
    """
    idx = int(np.argmax(prob_vec))
    conf = float(prob_vec[idx])
    return MODEL_PROB_LABELS[idx], conf

# -----------------------------
# Use your pipeline's models
# -----------------------------
def predict_all_models(pipe, headline: str, body: Optional[str] = None):
    txt_h = (headline or "").strip()
    txt_b = (body or "") if body else ""

    # FinBERT
    fb_head = pipe.finbert_probs(txt_h, temp=pipe.FB_TEMP)
    fb_body = pipe.finbert_probs(txt_b or txt_h, temp=pipe.FB_TEMP)
    fb = pipe.FB_HEAD_W * fb_head + (1.0 - pipe.FB_HEAD_W) * fb_body

    # RoBERTa (head+body if available)
    rb_head = pipe.roberta_probs(txt_h)
    rb_body = pipe.roberta_probs(txt_b or txt_h)
    rb = getattr(pipe, "RB_HEAD_W", 1.0) * rb_head + (1.0 - getattr(pipe, "RB_HEAD_W", 1.0)) * rb_body

    # Lexicons
    vd = pipe.vader_probs(txt_h)
    tb = pipe.textblob_probs(txt_b or txt_h)

    # Ensemble (no Alpha vector for these datasets)
    ens_label, ens_conf, _ = pipe.vote_probs(
        txt_h, txt_b, alpha_vec=None, weights=pipe.WEIGHTS_BASE, neutral_cap=pipe.NEUTRAL_CAP
    )

    fb_label, fb_conf = pred_from_probs(fb)
    rb_label, rb_conf = pred_from_probs(rb)
    vd_label, vd_conf = pred_from_probs(vd)
    tb_label, tb_conf = pred_from_probs(tb)

    return {
        "ensemble_label": ens_label,
        "ensemble_conf": ens_conf,
        "finbert_label": fb_label,
        "finbert_conf": fb_conf,
        "roberta_label": rb_label,
        "roberta_conf": rb_conf,
        "vader_label": vd_label,
        "vader_conf": vd_conf,
        "textblob_label": tb_label,
        "textblob_conf": tb_conf,
    }

# -----------------------------
# Evaluation utilities
# -----------------------------
def evaluate_rows(rows: List[Dict], truth_key: str, pred_key: str):
    y_true = [r[truth_key] for r in rows]
    y_pred = [r[pred_key] for r in rows]
    cm = confusion_matrix(y_true, y_pred, labels=EVAL_LABELS)
    rep = classification_report(y_true, y_pred, labels=EVAL_LABELS, zero_division=0, output_dict=True)
    kappa = cohen_kappa_score(y_true, y_pred, labels=EVAL_LABELS)
    return {
        "labels": EVAL_LABELS,
        "confusion_matrix": cm.tolist(),
        "cohen_kappa": round(float(kappa), 3),
        "macro_f1": round(float(rep["macro avg"]["f1-score"]), 3),
        "weighted_f1": round(float(rep["weighted avg"]["f1-score"]), 3),
        "per_class_f1": {c: round(float(rep[c]["f1-score"]), 3) for c in EVAL_LABELS},
        "support": {c: int(sum(1 for v in y_true if v == c)) for c in EVAL_LABELS},
    }

def print_eval(name: str, metrics: Dict):
    print(f"\n=== {name} ===")
    print("labels:", metrics["labels"])
    print("confusion:\n", np.array(metrics["confusion_matrix"]))
    print(
        "kappa:",
        metrics["cohen_kappa"],
        " macro_F1:",
        metrics["macro_f1"],
        " weighted_F1:",
        metrics["weighted_f1"],
    )
    print("per-class F1:", metrics["per_class_f1"], " support:", metrics["support"])

# -----------------------------
# Dataset loaders
# -----------------------------
def load_financial_phrasebank(config: Optional[str] = None):
    load_dataset = _safe_import_datasets()
    tries = ([( "financial_phrasebank", config )] if config else [
        ("financial_phrasebank", "sentences_allagree"),
        ("financial_phrasebank", "all"),
        ("financial_phrasebank", None),
    ])
    last_err = None
    for name, cfg in tries:
        try:
            ds = load_dataset(name, cfg) if cfg else load_dataset(name)
            split = "train" if "train" in ds else list(ds.keys())[0]
            records = []
            for ex in ds[split]:
                txt = ex.get("sentence") or ex.get("text") or ""
                lbl = ex.get("label") or ex.get("polarity") or ex.get("sentiment")
                records.append({"text": txt, "label": coerce_label(lbl)})
            return records
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load Financial PhraseBank. Last error: {last_err}")

def load_fiqa_from_hf(fiqa_id: Optional[str] = None, fiqa_split: Optional[str] = None):
    load_dataset = _safe_import_datasets()
    candidates = []
    if fiqa_id:
        candidates.append((fiqa_id, None))
    candidates += [
        ("TheFinAI/fiqa-sentiment-classification", None),
        ("ChanceFocus/fiqa-sentiment-classification", None),
        ("LLukas22/fiqa", None),
        ("SALT-NLP/FLUE-FiQA", None),
    ]
    last_err = None
    for name, cfg in candidates:
        try:
            ds = load_dataset(name, cfg) if cfg else load_dataset(name)
            split = (
                fiqa_split
                or ("validation" if "validation" in ds else
                    "valid" if "valid" in ds else
                    "test" if "test" in ds else
                    "train")
            )
            if split not in ds:
                split = list(ds.keys())[0]
            recs = []
            for ex in ds[split]:
                txt = (
                    ex.get("text")
                    or ex.get("sentence")
                    or ex.get("headline")
                    or ex.get("title")
                    or ex.get("news_title")
                    or ex.get("question_title")
                    or ""
                )
                raw_lbl = (
                    ex.get("label")
                    or ex.get("sentiment")
                    or ex.get("sentiment_label")
                    or ex.get("y")
                    or ex.get("score")
                )
                recs.append({"text": txt, "label": coerce_label(raw_lbl)})
            if recs:
                return recs
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load any FiQA mirror. Last error: {last_err}")

def load_fiqa_from_csv(path: str):
    p = pathlib.Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"FiQA CSV not found: {p}")
    df = pd.read_csv(p)
    # Accept common FiQA-style columns: sentence/text + label or numeric score
    text_col = None
    for c in ["text", "sentence", "headline", "title", "news_title", "question_title"]:
        if c in df.columns:
            text_col = c; break
    if text_col is None:
        raise RuntimeError("FiQA CSV must have one of: text|sentence|headline|title|news_title|question_title")
    label_col = None
    for c in ["label", "sentiment", "sentiment_label", "y", "score"]:
        if c in df.columns:
            label_col = c; break
    if label_col is None:
        raise RuntimeError("FiQA CSV must have a label/sentiment or score column: label|sentiment|sentiment_label|y|score")

    records = []
    for _, row in df.iterrows():
        txt = str(row[text_col]) if pd.notna(row[text_col]) else ""
        lbl = row[label_col]
        records.append({"text": txt, "label": coerce_label(lbl)})
    return records

def load_fiqa(fiqa_id: Optional[str], fiqa_split: Optional[str], fiqa_csv: Optional[str]):
    # Prefer CSV if provided; else HuggingFace
    if fiqa_csv:
        return load_fiqa_from_csv(fiqa_csv)
    return load_fiqa_from_hf(fiqa_id=fiqa_id, fiqa_split=fiqa_split)

# -----------------------------
# Benchmark runners
# -----------------------------
def evaluate_dataset(
    pipe,
    name: str,
    items: List[Dict],
    max_n: Optional[int] = 5000,
    outdir: str = "benchmarks",
    seed: int = 13,
):
    """
    Shuffle the dataset (reproducibly) before taking the first max_n examples.
    This prevents biased slices when datasets are ordered by label/source.
    """
    outp = pathlib.Path(outdir); outp.mkdir(parents=True, exist_ok=True)

    # Reproducible shuffle before truncation
    rng = np.random.RandomState(seed)
    items_shuf = list(items)
    rng.shuffle(items_shuf)

    use_items = items_shuf if (not max_n or len(items_shuf) <= max_n) else items_shuf[:max_n]

    rows = []
    for i, ex in enumerate(use_items, 1):
        t = (ex.get("text") or "").strip()
        if not t:
            continue
        pred = predict_all_models(pipe, headline=t, body=None)
        rows.append({"text": t, "true": ex["label"], **pred})
        if i % 100 == 0:
            print(f"[{name}] scored {i}/{len(use_items)}")

    if not rows:
        print(f"[{name}] No rows to evaluate.")
        return

    results = {
        "ensemble": evaluate_rows(rows, "true", "ensemble_label"),
        "finbert":  evaluate_rows(rows, "true", "finbert_label"),
        "roberta":  evaluate_rows(rows, "true", "roberta_label"),
        "vader":    evaluate_rows(rows, "true", "vader_label"),
        "textblob": evaluate_rows(rows, "true", "textblob_label"),
        "n": len(rows)
    }

    metrics_path = outp / f"{name}_metrics.json"
    rows_path    = outp / f"{name}_preds.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(rows_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # Append a compact summary row for quick comparison across runs
    summary_path = outp / "benchmark_summary.csv"
    summary_row = {
        "dataset": name,
        "n": results["n"],
        "ensemble_weighted_f1": results["ensemble"]["weighted_f1"],
        "roberta_weighted_f1": results["roberta"]["weighted_f1"],
        "finbert_weighted_f1": results["finbert"]["weighted_f1"],
        "vader_weighted_f1": results["vader"]["weighted_f1"],
        "textblob_weighted_f1": results["textblob"]["weighted_f1"],
        "ensemble_kappa": results["ensemble"]["cohen_kappa"],
        "preset_note": getattr(pipe, "NEUTRAL_CAP", None),
        "seed": seed,
        "max_n": max_n,
    }
    write_header = not summary_path.exists()
    with open(summary_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        if write_header: w.writeheader()
        w.writerow(summary_row)

    print(f"\nSaved {name} benchmark: {metrics_path} and {rows_path}")
    for key in ["ensemble","finbert","roberta","vader","textblob"]:
        print_eval(f"{name.upper()} — {key}", results[key])

def run_benchmark(
    pipe,
    max_n: int,
    phrasebank_config: Optional[str],
    fiqa_id: Optional[str],
    fiqa_split: Optional[str],
    fiqa_csv: Optional[str],
    seed: int = 13,
):
    print("\n=== Loading Financial PhraseBank ===")
    fpb = load_financial_phrasebank(config=phrasebank_config)
    print(f"Loaded Financial PhraseBank: {len(fpb)} rows")
    evaluate_dataset(pipe, "phrasebank", fpb, max_n=max_n, seed=seed)

    print("\n=== Loading FiQA ===")
    fiqa = load_fiqa(fiqa_id=fiqa_id, fiqa_split=fiqa_split, fiqa_csv=fiqa_csv)
    print(f"Loaded FiQA: {len(fiqa)} rows")
    evaluate_dataset(pipe, "fiqa", fiqa, max_n=max_n, seed=seed)

# -----------------------------
# Optional: evaluate AV vs human labels on your CSV
# -----------------------------
def evaluate_av_on_csv(pipe, csv_path: str, label_column: str = "ground_truth"):
    p = pathlib.Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(p)
    if label_column not in df.columns:
        raise RuntimeError(f"CSV must contain a '{label_column}' column with human labels.")

    def map_av(lbl: str) -> str:
        s = (str(lbl) or "").lower()
        if "bear" in s or "negative" in s:
            return _NEG
        if "bull" in s or "positive" in s:
            return _POS
        return _NEU

    df["alpha_coarse"] = df["alpha_label"].map(map_av)
    rows = df.to_dict("records")

    av_metrics   = evaluate_rows(rows, label_column, "alpha_coarse")
    ens_metrics  = evaluate_rows(rows, label_column, "pred_label")

    print("\n=== AV on your CSV (with human labels) ===")
    print_eval("AV vs ground truth", av_metrics)
    print("\n=== Ensemble on your CSV (with human labels) ===")
    print_eval("Ensemble vs ground truth", ens_metrics)

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark financial sentiment pipeline on FiQA & Financial PhraseBank."
    )
    parser.add_argument("--pipeline-path", required=True, help="Path to your sentiment pipeline .py file")
    parser.add_argument("--benchmark", action="store_true", help="Run FiQA & Financial PhraseBank benchmarks")
    parser.add_argument("--eval-av", type=str, default=None, help="Path to alpha_sentiment.csv with a 'ground_truth' column")
    parser.add_argument("--max-n", type=int, default=5000, help="Maximum rows per dataset to score (e.g., 500 or 1000 for quick runs)")
    parser.add_argument("--phrasebank-config", type=str, default=None, help="Financial PhraseBank config (e.g., 'sentences_allagree', 'all')")
    parser.add_argument("--fiqa-id", type=str, default=None, help="Override FiQA dataset HF ID (e.g., 'TheFinAI/fiqa-sentiment-classification')")
    parser.add_argument("--fiqa-split", type=str, default=None, help="Specific split for the chosen FiQA dataset (e.g., 'validation', 'test', 'train')")
    parser.add_argument("--fiqa-csv", type=str, default=None, help="Local FiQA CSV fallback (must include text/sentence and label|score columns)")
    # Domain presets + overrides
    parser.add_argument("--preset", type=str, choices=["news", "fiqa"], default=None, help="Apply domain preset before benchmarking")
    parser.add_argument("--neutral-cap", type=float, default=None, help="Override pipeline NEUTRAL_CAP (e.g., 0.28)")
    parser.add_argument("--soften-tau", type=float, default=None, help="Override pipeline SOFTEN_TAU (e.g., 0.98)")
    parser.add_argument("--override-weights", type=str, default=None,
                        help='JSON dict to override WEIGHTS_BASE, e.g. \'{"roberta":0.38,"vader":0.03,"textblob":0.03}\'')
    # Reproducible shuffling
    parser.add_argument("--seed", type=int, default=13, help="Random seed for dataset shuffling")

    args = parser.parse_args()

    pipe = load_pipeline_module(args.pipeline_path)

    # Apply preset first, then individual overrides
    apply_preset(pipe, args.preset)
    apply_overrides(pipe, args.neutral_cap, args.soften_tau, args.override_weights)

    if args.benchmark:
        run_benchmark(
            pipe,
            max_n=args.max_n,
            phrasebank_config=args.phrasebank_config,
            fiqa_id=args.fiqa_id,
            fiqa_split=args.fiqa_split,
            fiqa_csv=args.fiqa_csv,
            seed=args.seed,
        )

    if args.eval_av:
        evaluate_av_on_csv(pipe, args.eval_av)

    if not args.benchmark and not args.eval_av:
        print("Nothing to do. Use --benchmark and/or --eval-av <csv_path>.")

if __name__ == "__main__":
    main()
