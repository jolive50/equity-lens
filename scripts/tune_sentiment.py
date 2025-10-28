# tune_pipeline.py
# Hyperparameter tuner for your sentiment pipeline.
# Stages:
#   1) Alpha sweep: try alpha weight across [alpha_min, alpha_max]
#   2) Random jitter: fine-tune non-Alpha weights & knobs while holding Alpha fixed
#
# Notes:
# - Matches the benchmark script's loaders and label handling.
# - Reproducible shuffling BEFORE truncating (--max-n).
# - Writes JSONL of trials if --out is given.

import argparse
import importlib.util
import json
import pathlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

# -----------------------------
# Label helpers (match benchmark)
# -----------------------------
_POS = "positive"; _NEU = "neutral"; _NEG = "negative"
EVAL_LABELS = [_NEG, _NEU, _POS]
MODEL_PROB_LABELS = [_POS, _NEU, _NEG]

def coerce_label(x) -> str:
    if x is None:
        return _NEU
    s = str(x).strip().lower()
    if s in {"2","pos","positive","bullish"}: return _POS
    if s in {"1","neu","neutral"}:            return _NEU
    if s in {"0","neg","negative","bearish"}: return _NEG
    try:
        v = float(s)
        if v > 0.05: return _POS
        if v < -0.05: return _NEG
        return _NEU
    except Exception:
        if "pos" in s: return _POS
        if "neg" in s: return _NEG
        return _NEU

# -----------------------------
# Dynamic import of your pipeline module
# -----------------------------
def load_pipeline_module(pipeline_path: str):
    p = pathlib.Path(pipeline_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Pipeline script not found: {p}")
    spec = importlib.util.spec_from_file_location("sent_pipeline", str(p))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader, "Could not prepare module loader"
    spec.loader.exec_module(mod)
    required = [
        "vote_probs","finbert_probs","roberta_probs","vader_probs","textblob_probs",
        "WEIGHTS_BASE","NEUTRAL_CAP","FB_TEMP","FB_HEAD_W"
    ]
    for name in required:
        if not hasattr(mod, name):
            raise RuntimeError(f"Pipeline missing: {name}")
    if not hasattr(mod, "RB_HEAD_W"): setattr(mod, "RB_HEAD_W", 1.0)
    if not hasattr(mod, "SOFTEN_TAU"): setattr(mod, "SOFTEN_TAU", 1.0)
    return mod

# -----------------------------
# Presets / overrides (optional)
# -----------------------------
def apply_preset(pipe, preset: Optional[str]):
    if not preset: return
    if preset.lower() == "fiqa":
        try:
            pipe.NEUTRAL_CAP = 0.28
            pipe.SOFTEN_TAU  = 0.98
            w = dict(pipe.WEIGHTS_BASE)
            w.update({"roberta":0.38, "vader":0.03, "textblob":0.03})
            s = sum(w.values()); pipe.WEIGHTS_BASE = {k: v/s for k, v in w.items()}
        except Exception:
            pass
    # "news": keep as-is

# -----------------------------
# Loaders (self-contained; no HF dependency if you pass CSVs)
# -----------------------------
def _safe_import_datasets():
    try:
        from datasets import load_dataset  # type: ignore
        return load_dataset
    except Exception as e:
        raise RuntimeError("Install datasets: `pip install datasets`") from e

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
    if fiqa_id: candidates.append((fiqa_id, None))
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
                    "valid"      if "valid" in ds else
                    "test"       if "test" in ds else
                    "train")
            )
            if split not in ds:
                split = list(ds.keys())[0]
            recs = []
            for ex in ds[split]:
                txt = (ex.get("text") or ex.get("sentence") or ex.get("headline")
                       or ex.get("title") or ex.get("news_title") or ex.get("question_title") or "")
                raw = (ex.get("label") or ex.get("sentiment") or ex.get("sentiment_label")
                       or ex.get("y") or ex.get("score"))
                recs.append({"text": txt, "label": coerce_label(raw)})
            if recs:
                return recs
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load any FiQA mirror. Last error: {last_err}")

# -----------------------------
# Scoring helpers
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

def predict_all_models(pipe, headline: str):
    txt_h = (headline or "").strip()
    fb_head = pipe.finbert_probs(txt_h, temp=pipe.FB_TEMP)
    fb = pipe.FB_HEAD_W * fb_head + (1.0 - pipe.FB_HEAD_W) * fb_head
    rb_head = pipe.roberta_probs(txt_h)
    rb = getattr(pipe, "RB_HEAD_W", 1.0) * rb_head + (1.0 - getattr(pipe, "RB_HEAD_W", 1.0)) * rb_head
    vd = pipe.vader_probs(txt_h)
    tb = pipe.textblob_probs(txt_h)
    ens_label, ens_conf, _ = pipe.vote_probs(
        txt_h, None, alpha_vec=None, weights=pipe.WEIGHTS_BASE, neutral_cap=pipe.NEUTRAL_CAP
    )
    # simple map from probs already done by vote_probs
    return {"ensemble_label": ens_label}

def score_dataset(pipe, items: List[Dict], max_n: int, seed: int) -> Dict:
    rng = np.random.RandomState(seed)
    items_shuf = list(items); rng.shuffle(items_shuf)
    use = items_shuf if (not max_n or len(items_shuf) <= max_n) else items_shuf[:max_n]
    rows = []
    for ex in use:
        t = (ex.get("text") or "").strip()
        if not t: continue
        pred = predict_all_models(pipe, t)
        rows.append({"text": t, "true": ex["label"], **pred})
    return evaluate_rows(rows, "true", "ensemble_label")

def score_all(pipe, datasets: List[str], max_n: int, seed: int,
              phrasebank_config: Optional[str], fiqa_id: Optional[str], fiqa_split: Optional[str]) -> Dict:
    metrics = {}
    if "phrasebank" in datasets:
        fpb = load_financial_phrasebank(config=phrasebank_config)
        metrics["phrasebank"] = score_dataset(pipe, fpb, max_n=max_n, seed=seed)
    if "fiqa" in datasets:
        fiqa = load_fiqa_from_hf(fiqa_id=fiqa_id, fiqa_split=fiqa_split)
        metrics["fiqa"] = score_dataset(pipe, fiqa, max_n=max_n, seed=seed)
    return metrics

def avg_weighted_f1(metrics: Dict) -> float:
    vals = [m["weighted_f1"] for m in metrics.values() if m and "weighted_f1" in m]
    return float(np.mean(vals)) if vals else 0.0

# -----------------------------
# Weight utilities
# -----------------------------
def renorm(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(d.values()); 
    if s <= 0: raise ValueError("sum of weights must be > 0")
    return {k: v/s for k, v in d.items()}

def set_alpha_weight(pipe, alpha_w: float):
    # Distribute (1 - alpha_w) across the remaining models proportionally
    w = dict(pipe.WEIGHTS_BASE)
    others = {k: v for k, v in w.items() if k != "alpha"}
    s_others = sum(others.values()) or 1.0
    scaled = {k: (v / s_others) * (1.0 - alpha_w) for k, v in others.items()}
    w_new = {"alpha": alpha_w, **scaled}
    pipe.WEIGHTS_BASE = w_new

def jitter_non_alpha_weights(pipe, rng: np.random.RandomState, jitter: float, alpha_fixed: float):
    base = dict(pipe.WEIGHTS_BASE)
    others = {k: v for k, v in base.items() if k != "alpha"}
    vec = np.array(list(others.values()))
    noise = rng.normal(0.0, jitter, size=vec.shape)
    vec = np.maximum(1e-6, vec + noise)
    vec = (vec / vec.sum()) * (1.0 - alpha_fixed)
    w_new = {"alpha": alpha_fixed}
    for k, val in zip(others.keys(), vec):
        w_new[k] = float(val)
    pipe.WEIGHTS_BASE = w_new

# -----------------------------
# Main tuning routine
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Tune sentiment pipeline with alpha sweep + jitter.")
    ap.add_argument("--pipeline-path", required=True, help="Path to your pipeline .py")
    ap.add_argument("--benchmark-path", default=None, help="(unused; kept for signature compatibility)")
    ap.add_argument("--trials", type=int, default=60, help="Random jitter trials after alpha sweep")
    ap.add_argument("--max-n", type=int, default=1000, help="Max examples per dataset")
    ap.add_argument("--datasets", type=str, default="fiqa,phrasebank", help="Comma list: fiqa,phrasebank")
    ap.add_argument("--seed", type=int, default=13, help="Random seed for shuffling and jitter")
    ap.add_argument("--preset", type=str, choices=["news","fiqa"], default=None, help="Apply domain preset before tuning")
    ap.add_argument("--weight-jitter", type=float, default=0.02, help="Stddev for non-Alpha weight noise")

    # Alpha sweep controls
    ap.add_argument("--alpha-min", type=float, default=0.30)
    ap.add_argument("--alpha-max", type=float, default=0.40)
    ap.add_argument("--alpha-steps", type=int, default=6)
    ap.add_argument("--skip-alpha", action="store_true", help="Skip alpha sweep, keep pipeline alpha as-is")

    # Optional knob ranges (light search)
    ap.add_argument("--fb-temp", type=str, default="2.0,2.4", help="Range for FB_TEMP (min,max)")
    ap.add_argument("--rb-head-w", type=str, default="0.70,0.90", help="Range for RB_HEAD_W (min,max)")
    ap.add_argument("--neutral-cap", type=str, default="0.26,0.34", help="Range for NEUTRAL_CAP (min,max)")
    ap.add_argument("--soften-tau", type=str, default="0.95,1.00", help="Range for SOFTEN_TAU (min,max)")

    ap.add_argument("--phrasebank-config", type=str, default=None)
    ap.add_argument("--fiqa-id", type=str, default=None)
    ap.add_argument("--fiqa-split", type=str, default=None)

    ap.add_argument("--out", type=str, default=None, help="Write JSONL of trials here")

    args = ap.parse_args()
    rng = np.random.RandomState(args.seed)

    pipe = load_pipeline_module(args.pipeline_path)
    apply_preset(pipe, args.preset)

    datasets = [s.strip().lower() for s in args.datasets.split(",") if s.strip()]
    if not datasets:
        raise RuntimeError("No datasets selected.")

    def evaluate_current(stage: str, trial: int, note: Dict) -> Dict:
        metrics = score_all(pipe, datasets, max_n=args.max_n, seed=args.seed,
                            phrasebank_config=args.phrasebank_config,
                            fiqa_id=args.fiqa_id, fiqa_split=args.fiqa_split)
        result = {
            "stage": stage,
            "trial": trial,
            "weights": dict(pipe.WEIGHTS_BASE),
            "neutral_cap": float(getattr(pipe, "NEUTRAL_CAP", 0.31)),
            "soften_tau": float(getattr(pipe, "SOFTEN_TAU", 1.0)),
            "fb_temp": float(getattr(pipe, "FB_TEMP", 2.2)),
            "rb_head_w": float(getattr(pipe, "RB_HEAD_W", 0.85)),
            "metrics": metrics,
            "avg_weighted_f1": avg_weighted_f1(metrics),
            **note,
        }
        return result

    out_f = open(args.out, "a", encoding="utf-8") if args.out else None
    def log_write(obj):
        if out_f:
            out_f.write(json.dumps(obj) + "\n")
            out_f.flush()

    best = None

    # -------- Stage 1: Alpha sweep --------
    if not args.skip_alpha:
        alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
        # preserve relative shares of non-alpha weights
        base_non_alpha = {k: v for k, v in pipe.WEIGHTS_BASE.items() if k != "alpha"}
        base_non_alpha = renorm(base_non_alpha)
        for i, a in enumerate(alphas, 1):
            # Set alpha and scale others
            scaled = {k: (v * (1.0 - a)) for k, v in base_non_alpha.items()}
            pipe.WEIGHTS_BASE = {"alpha": float(a), **scaled}

            res = evaluate_current("alpha", i, {"alpha": float(a)})
            log_write(res)
            if (best is None) or (res["avg_weighted_f1"] > best["avg_weighted_f1"]):
                best = res
                print("\n=== BEST (Alpha sweep) ===")
                print(json.dumps(best, indent=2))

    # Hold alpha fixed at best (or existing if skipped)
    alpha_fixed = float(best["alpha"]) if (best and "alpha" in best) else float(pipe.WEIGHTS_BASE.get("alpha", 0.0))
    set_alpha_weight(pipe, alpha_fixed)

    # -------- Stage 2: Random jitter over other weights + knobs --------
    fb_lo, fb_hi = [float(x) for x in args.fb_temp.split(",")]
    rb_lo, rb_hi = [float(x) for x in args.rb_head_w.split(",")]
    nc_lo, nc_hi = [float(x) for x in args.neutral_cap.split(",")]
    st_lo, st_hi = [float(x) for x in args.soften_tau.split(",")]

    # Start from the current best if we have one
    if best:
        pipe.NEUTRAL_CAP = best["neutral_cap"]
        pipe.SOFTEN_TAU  = best["soften_tau"]
        pipe.FB_TEMP     = best.get("fb_temp", getattr(pipe, "FB_TEMP", 2.2))
        pipe.RB_HEAD_W   = best.get("rb_head_w", getattr(pipe, "RB_HEAD_W", 0.85))
        pipe.WEIGHTS_BASE = dict(best["weights"])

    for t in range(1, args.trials + 1):
        # Jitter non-alpha weights
        jitter_non_alpha_weights(pipe, rng, args.weight_jitter, alpha_fixed=alpha_fixed)
        # Sample knobs
        pipe.FB_TEMP   = float(rng.uniform(fb_lo, fb_hi))
        pipe.RB_HEAD_W = float(rng.uniform(rb_lo, rb_hi))
        pipe.NEUTRAL_CAP = float(rng.uniform(nc_lo, nc_hi))
        pipe.SOFTEN_TAU  = float(rng.uniform(st_lo, st_hi))

        res = evaluate_current("jitter", t, {"alpha": alpha_fixed})
        log_write(res)
        if best is None or res["avg_weighted_f1"] > best["avg_weighted_f1"]:
            best = res
            print("\n=== NEW BEST (Jitter) ===")
            print(json.dumps(best, indent=2))

    if out_f:
        out_f.close()

    print("\n=== FINAL BEST ===")
    print(json.dumps(best, indent=2))
    print("\nTip: copy these into your pipeline constants (WEIGHTS_BASE / NEUTRAL_CAP / SOFTEN_TAU / FB_TEMP / RB_HEAD_W).")

if __name__ == "__main__":
    main()
