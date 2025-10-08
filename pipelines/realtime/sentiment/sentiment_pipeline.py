# -*- coding: utf-8 -*-
# Alpha Vantage News + Ensemble Sentiment (FinBERT + RoBERTa + VADER + TextBlob + AV)
# - Agreement-aware reweighting gate (+ negative gate)
# - FinBERT temp smoothing + headline-first mix
# - RoBERTa head+body blend + confidence-gated boost
# - Neutral cap + ENTROPY-AWARE neutrality (now respects passed neutral_cap)
# - Robust scraping with og:title fallback, junk-headline guard, and FETCH CACHE
# - Alpha Vantage time format fix + retry/backoff + proper pagination
# - GPU/CPU auto-device + AMP for speed
# - FIX: tokenizer fallback to avoid "chat_template TypeError" in some transformers versions
# - NEW: requests.Session with retry, rate-limit friendly sleeps
# - NEW: ALPHAVANTAGE_KEY read from environment

# ----------------- Config -----------------
import os

ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")
if not ALPHAVANTAGE_KEY or not ALPHAVANTAGE_KEY.strip():
    raise RuntimeError("Missing Alpha Vantage key. Set environment variable ALPHAVANTAGE_KEY.")

TICKERS   = "AAPL"
TOPICS    = "technology,earnings"
TIME_FROM = "2025-08-24T00:00:00Z"
TIME_TO   = "2025-09-20T23:59:59Z"
LIMIT     = 50
PAGES     = 1

# Tuned base weights (sum≈1.0)
WEIGHTS_BASE = {
    "alpha":    0.30,
    "finbert":  0.25779860648595176,
    "roberta":  0.41089259537549316,
    "vader":    0.015471796550132269,
    "textblob": 0.015837001588422658,
}

# Tuned ensemble knobs
NEUTRAL_CAP = 0.32
SOFTEN_TAU  = 0.9608949504565845
FB_TEMP     = 2.003765108032388
FB_HEAD_W   = 0.92
RB_HEAD_W   = 0.7716667565409939  # headline-first blend for RoBERTa

# RoBERTa confidence boost
RB_POS_BOOST_T = 0.70
RB_NEG_BOOST_T = 0.70
RB_BOOST_ADD   = 0.09

# Agreement gate thresholds
A_MIN = 0.50; R_MIN = 0.58; F_MIN = 0.60
USE_NEG_GATE = True
A_NEG_MIN = 0.50; R_NEG_MIN = 0.58; F_POS_MIN = 0.60

# Logging
PRINT_PER_MODEL  = True
PRINT_GATE_DIAGS = True
SAVE_CSV_PATH    = "alpha_sentiment.csv"

# ---------- Imports & setup ----------
import re, time, csv, requests, numpy as np, torch
from datetime import datetime
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from requests.adapters import HTTPAdapter, Retry
from pipelines.realtime.sentiment.agent import run_sentiment_agent


os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"

# ---- Torch device setup ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
try:
    if DEVICE == "cuda":
        torch.backends.cudnn.benchmark = True
except Exception:
    pass

LABELS = ["positive","neutral","negative"]
JUNK_HEAD_PAT = re.compile(r"^\s*(member\s*sign\s*in|sign\s*in|subscribe|log\s*in)\s*$", re.I)
BOILERPLATE_PATTERNS = [
    r"Reporting by .*?;.*?Editing by", r"Our Standards.*", r"Read more:", r"\(Reuters\)",
    r"Sign up", r"©", r"All rights reserved", r"Follow us on", r"Related", r"Trending"
]

# -------------- Utilities --------------
def norm3(a):
    a = np.maximum(np.array(a, dtype=float), 0)
    s = a.sum()
    return a / s if s > 0 else np.array([1/3,1/3,1/3], dtype=float)

def polarity_to_probs(p):
    pos = max(0.0, p)
    neg = max(0.0, -p)
    neu = 1.0 - (pos + neg)
    return norm3([pos, neu, neg])

def soften_neutral(p, tau=1.0):
    return norm3(tau*np.array(p, float) + (1.0 - tau)*np.array([1/3,1/3,1/3], float))

def _fmt(p): return f"pos={p[0]:.3f} neu={p[1]:.3f} neg={p[2]:.3f}"

def _entropy(p):
    p = np.clip(np.array(p, float), 1e-9, 1.0)
    return float(-(p * np.log(p)).sum())

# Dynamic neutrality controls
NEUTRAL_CAP_BASE = NEUTRAL_CAP
NEUTRAL_CAP_MIN  = 0.23
NEUTRAL_CAP_MAX  = 0.37
ENTROPY_BETA     = 1.0
MIN_POSNEG_EDGE  = 0.58

def dynamic_neutral_cap(combined, base_cap):
    """Entropy-aware neutral threshold that respects the caller's base_cap."""
    h = _entropy(combined)
    h_norm = min(h / np.log(3.0), 1.0)  # 0..1
    low, high = NEUTRAL_CAP_MIN, NEUTRAL_CAP_MAX
    cap = low*(1.0 - h_norm)**ENTROPY_BETA + high*(h_norm**ENTROPY_BETA)
    return float(0.5*cap + 0.5*base_cap)

def _to_av_time(s):
    """
    Accepts ISO-like strings ('2025-08-24T00:00:00Z', '2025-08-24 00:00')
    or already-correct AV strings ('YYYYMMDDTHHMM'). Returns 'YYYYMMDDTHHMM'.
    """
    if not s:
        return None
    s = s.strip()
    if re.fullmatch(r"\d{8}T\d{4}", s):
        return s
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y%m%dT%H%M")
        except ValueError:
            pass
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 12:
        return f"{digits[:8]}T{digits[8:12]}"
    raise ValueError(f"Unrecognized time format for Alpha Vantage: {s}")

# -------------- Scraping --------------
_FETCH_CACHE = {}  # fetch cache

def clean_paragraphs(paragraphs):
    cleaned = []
    for p in paragraphs:
        txt = p.get_text(" ", strip=True)
        if not txt: continue
        if any(re.search(pat, txt, flags=re.I) for pat in BOILERPLATE_PATTERNS):
            continue
        cleaned.append(txt)
    return cleaned

def fetch_article(url: str):
    """Best-effort scrape of headline + article body, with og:title fallback."""
    if not url:
        return "", ""
    if url in _FETCH_CACHE:
        return _FETCH_CACHE[url]
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urlopen(req, timeout=20).read()
        soup = BeautifulSoup(html, "html.parser")
        h = soup.find("h1")
        headline = h.get_text(" ", strip=True) if h else ""
        if not headline:
            og = soup.find("meta", attrs={"property": "og:title"})
            if og and og.get("content"): headline = og["content"].strip()
        art = soup.find("article")
        ps = art.find_all("p") if art else soup.find_all("p")
        ps = clean_paragraphs(ps)
        body = " ".join(ps)
        _FETCH_CACHE[url] = (headline, body)
        return headline, body
    except Exception:
        _FETCH_CACHE[url] = ("", "")
        return "", ""

# -------------- Models --------------
def load_tok(model_name: str):
    """Prefer fast tokenizer; fall back to slow to avoid chat_template TypeError on some versions."""
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except Exception:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)

# FinBERT
_FINBERT = "ProsusAI/finbert"
_fin_tok = load_tok(_FINBERT)
_fin_mdl = AutoModelForSequenceClassification.from_pretrained(
    _FINBERT, use_safetensors=True
).to(DEVICE).eval()

def _softmax_temp_logits(logits, t=1.0):
    z = logits / t
    z = z - z.max(dim=-1, keepdim=True).values
    ez = torch.exp(z)
    p = ez / ez.sum(dim=-1, keepdim=True)
    return p.detach().cpu().numpy()[0]

def finbert_probs(text: str, temp=1.0):
    t = (text or "").strip()
    if not t: return np.array([1/3,1/3,1/3], float)
    inputs = _fin_tok(t, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
        logits = _fin_mdl(**inputs).logits
        if temp and temp > 1.0:
            return _softmax_temp_logits(logits, t=temp) # [pos,neu,neg]
        p = torch.softmax(logits, dim=-1)
        return p.detach().cpu().numpy()[0]

# RoBERTa (Cardiff)
_RB = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_rb_tok = load_tok(_RB)
_rb_mdl = AutoModelForSequenceClassification.from_pretrained(
    _RB, use_safetensors=True
).to(DEVICE).eval()

def roberta_probs(text: str):
    t = (text or "").strip()
    if not t: return np.array([1/3,1/3,1/3], float)
    inputs = _rb_tok(t[:512], return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
        logits = _rb_mdl(**inputs).logits
        p = torch.softmax(logits, dim=-1).detach().cpu().numpy()[0]  # [neg,neu,pos]
    return np.array([p[2], p[1], p[0]], float)  # -> [pos,neu,neg]

# VADER + TextBlob
_vader = SentimentIntensityAnalyzer()
def vader_probs(text: str):    return polarity_to_probs(_vader.polarity_scores(text or "")["compound"])
def textblob_probs(text: str): return polarity_to_probs(TextBlob(text or "").sentiment.polarity)

# -------------- Alpha Vantage mapping --------------
def alpha_probs(overall_label: str = None, overall_score: float = None):
    """
    Convert AV (label, score) to [pos,neu,neg].
    - Mildly amplifies small scores, keeps neutral mass.
    """
    lbl = (overall_label or "").strip().lower()
    if overall_score is not None:
        s = float(overall_score)
        neu = 0.32
        mag = min(abs(s) * 1.35, 0.72)
        pos = mag if s > 0 else 0.0
        neg = mag if s < 0 else 0.0
        return norm3([pos, neu, neg])
    if "bull" in lbl: return np.array([0.60, 0.30, 0.10])
    if "bear" in lbl: return np.array([0.10, 0.30, 0.60])
    return np.array([0.20, 0.60, 0.20])

# -------------- Agreement-aware reweighting --------------
def _agreement_adjust(weights, fb, rb, alpha_vec, tag=""):
    """
    Positive gate: Alpha & RoBERTa POS (confident) + FinBERT NEG (confident) → downweight FinBERT, boost AV/RoBERTa.
    Negative gate: Alpha & RoBERTa NEG + FinBERT POS → symmetric action.
    Also: confidence-gated RoBERTa boost (pos/neg) regardless of AV presence.
    """
    w = dict(weights)

    # --- Confidence-gated RoBERTa boost (independent of AV) ---
    r_pos = float(rb[0])
    r_neg = float(rb[2])
    boosted = False
    if r_pos >= RB_POS_BOOST_T:
        w["roberta"] = w.get("roberta", 0) + RB_BOOST_ADD
        boosted = True
    if r_neg >= RB_NEG_BOOST_T:
        w["roberta"] = w.get("roberta", 0) + RB_BOOST_ADD
        boosted = True
    if boosted:
        s = sum(w.values())
        if s > 0:
            for k in w: w[k] /= s
        if PRINT_GATE_DIAGS:
            print(f"[rb-boost {tag}] RoBERTa boost applied; weights now: {w}")

    # --- Positive consensus gate (requires AV) ---
    a_pos = float(alpha_vec[0]) if alpha_vec is not None else 0.0
    f_neg = float(fb[2])
    if PRINT_GATE_DIAGS:
        print(f"[gate inputs] {tag} alpha_vec: {alpha_vec} | roberta: {rb} | finbert: {fb}")
        print(f"[gate+ {tag}] a_pos={a_pos:.3f} (>= {A_MIN}) | r_pos={r_pos:.3f} (>= {R_MIN}) | f_neg={f_neg:.3f} (>= {F_MIN})")

    if (alpha_vec is not None and a_pos >= A_MIN) and (r_pos >= R_MIN) and (f_neg >= F_MIN):
        scale = 0.65   # shrink FinBERT
        add   = 0.150  # redistribute to alpha/roberta
        w["finbert"] = max(0.0, w.get("finbert", 0) * scale)
        w["alpha"]   = w.get("alpha", 0)   + add * 0.6
        w["roberta"] = w.get("roberta", 0) + add * 0.4
        s = sum(w.values())
        if s > 0:
            for k in w: w[k] /= s
        if PRINT_GATE_DIAGS:
            print(f"[gate+ {tag}] -> TRIGGERED; weights now: {w}")
        return w
    else:
        if PRINT_GATE_DIAGS:
            print(f"[gate+ {tag}] -> NOT triggered")

    # --- Negative consensus gate ---
    if USE_NEG_GATE and alpha_vec is not None:
        a_neg = float(alpha_vec[2])
        r_neg_gate = float(rb[2])
        f_pos = float(fb[0])
        if PRINT_GATE_DIAGS:
            print(f"[gate- {tag}] a_neg={a_neg:.3f} (>= {A_NEG_MIN}) | r_neg={r_neg_gate:.3f} (>= {R_NEG_MIN}) | f_pos={f_pos:.3f} (>= {F_POS_MIN})")
        if (a_neg >= A_NEG_MIN) and (r_neg_gate >= R_NEG_MIN) and (f_pos >= F_POS_MIN):
            scale = 0.65
            add   = 0.150
            w["finbert"] = max(0.0, w.get("finbert", 0) * scale)
            w["alpha"]   = w.get("alpha", 0)   + add * 0.6
            w["roberta"] = w.get("roberta", 0) + add * 0.4
            s = sum(w.values())
            if s > 0:
                for k in w: w[k] /= s
            if PRINT_GATE_DIAGS:
                print(f"[gate- {tag}] -> TRIGGERED; weights now: {w}")
            return w
        else:
            if PRINT_GATE_DIAGS:
                print(f"[gate- {tag}] -> NOT triggered")

    return w

# -------------- Ensemble ----------------
def vote_probs(headline: str, body: str, alpha_vec=None, weights=None, neutral_cap=NEUTRAL_CAP):
    """Weighted soft vote with optional Alpha vector & agreement-aware reweighting."""
    bad_head = JUNK_HEAD_PAT.match(headline or "")
    head_text = ((headline or "").strip() if (headline and not bad_head) else "") or (body[:180] if body else "")

    # FinBERT headline-first with temperature smoothing
    fb_head = finbert_probs(head_text, temp=FB_TEMP)
    fb_body = finbert_probs(body, temp=FB_TEMP) if body else fb_head
    fb = FB_HEAD_W*fb_head + (1.0 - FB_HEAD_W)*fb_body

    # RoBERTa head+body blend (headline-first)
    rb_head = roberta_probs(head_text)
    rb_body = roberta_probs(body) if body else rb_head
    rb = RB_HEAD_W*rb_head + (1.0 - RB_HEAD_W)*rb_body

    vd = vader_probs(head_text)                  # VADER on headline
    tb = textblob_probs(body or head_text)       # TextBlob on body if present

    if SOFTEN_TAU < 1.0:
        vd = soften_neutral(vd, SOFTEN_TAU)
        tb = soften_neutral(tb, SOFTEN_TAU)

    # Start with base weights, then apply gates/boosts
    w = dict(weights or WEIGHTS_BASE)
    if PRINT_GATE_DIAGS:
        print("weights(before):", w)
    w = _agreement_adjust(w, fb, rb, alpha_vec, tag="")
    if PRINT_GATE_DIAGS:
        print("weights(after): ", w)

    # Weighted sum
    vec = (w.get("finbert",0)*fb +
           w.get("roberta",0)*rb +
           w.get("vader",0)*vd +
           w.get("textblob",0)*tb)
    if alpha_vec is not None and "alpha" in w:
        vec = vec + w["alpha"]*np.array(alpha_vec, float)

    combined = norm3(vec)
    idx = int(np.argmax(combined))
    label = LABELS[idx]
    conf  = float(combined[idx])

    # Entropy-aware neutral decision (now uses the passed neutral_cap)
    dyn_cap = dynamic_neutral_cap(combined, base_cap=neutral_cap)
    if (label in ("positive","negative")) and (conf >= MIN_POSNEG_EDGE):
        pass
    else:
        if conf < dyn_cap:
            label, conf = "neutral", float(combined[1])

    breakdown = {
        "finbert": dict(zip(LABELS, map(float, fb))),
        "roberta": dict(zip(LABELS, map(float, rb))),
        "vader":   dict(zip(LABELS, map(float, vd))),
        "textblob":dict(zip(LABELS, map(float, tb))),
        "combined":dict(zip(LABELS, map(float, combined)))
    }
    return label, conf, breakdown

# -------------- Alpha Vantage client (with session + backoff + pagination) --------------
# Build a single session with retry policy to reuse connections and handle transient failures.
_session = requests.Session()
_retries = Retry(
    total=5,
    backoff_factor=0.8,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
_session.mount("https://", HTTPAdapter(max_retries=_retries))

ALPHAVANTAGE_RATE_SLEEP = 12.5  # ~5 requests/min for free tier

def av_news_sentiment(tickers=None, topics=None, time_from=None, time_to=None,
                      sort="LATEST", limit=50, page=1, max_retries=4):
    """
    Generator of AV news items with simple exponential backoff on API messages (429/Note).
    https://www.alphavantage.co/documentation/#newsapi
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "apikey": ALPHAVANTAGE_KEY,
        "sort": sort,
        "limit": min(int(limit), 1000),
        "page": int(page),  # <-- pagination now respected
    }
    if tickers:   params["tickers"]   = tickers
    if topics:    params["topics"]    = topics
    if time_from: params["time_from"] = _to_av_time(time_from)
    if time_to:   params["time_to"]   = _to_av_time(time_to)

    backoff = 0.7
    for attempt in range(max_retries):
        try:
            r = _session.get("https://www.alphavantage.co/query", params=params, timeout=25)
            r.raise_for_status()
            data = r.json()
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 1.8
            continue

        feed = data.get("feed", [])
        note = data.get("Note") or data.get("Information") or data.get("Error Message")
        if not feed and note:
            # Likely throttling / limit. Sleep more generously.
            if attempt == max_retries - 1:
                print(f"[AlphaVantage] {note}  (continuing with empty feed)")
                return
            time.sleep(max(ALPHAVANTAGE_RATE_SLEEP, backoff))
            backoff *= 1.8
            continue

        for item in feed:
            yield item
        return  # success; stop retries

# -------------- Batch analysis --------------
def analyze_alpha_news_batch(tickers=TICKERS, topics=TOPICS,
                             time_from=TIME_FROM, time_to=TIME_TO,
                             limit=LIMIT, pages=PAGES):
    rows = []
    seen_urls = set()
    for p in range(1, pages+1):
        for item in av_news_sentiment(tickers=tickers, topics=topics,
                                      time_from=time_from, time_to=time_to,
                                      limit=limit, page=p):
            url = item.get("url") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            alpha_label = item.get("overall_sentiment_label")
            alpha_score = item.get("overall_sentiment_score")
            alpha_vec   = alpha_probs(alpha_label, alpha_score)

            headline, body = fetch_article(url)

            # If scraped headline is junk/empty, use AV title
            if (not headline) or JUNK_HEAD_PAT.match(headline or ""):
                alt = item.get("title") or ""
                if alt: headline = alt

            # If body is tiny or blocked, append AV summary/content
            if (not body or len(body) < 600):
                fallback = " ".join([item.get("summary") or "", item.get("content") or ""]).strip()
                if fallback:
                    body = (body + "\n" if body else "") + fallback

            print("\n" + url)
            print("Headline:", (headline[:160] + "...") if headline and len(headline) > 160 else (headline or "(none)"))
            print("Body chars:", len(body or ""))

            label, conf, breakdown = vote_probs(
                headline, body, alpha_vec=alpha_vec, weights=WEIGHTS_BASE, neutral_cap=NEUTRAL_CAP
            )

            if PRINT_PER_MODEL:
                fb = [breakdown['finbert']['positive'], breakdown['finbert']['neutral'], breakdown['finbert']['negative']]
                rb = [breakdown['roberta']['positive'], breakdown['roberta']['neutral'], breakdown['roberta']['negative']]
                vd = [breakdown['vader']['positive'],   breakdown['vader']['neutral'],   breakdown['vader']['negative']]
                tb = [breakdown['textblob']['positive'],breakdown['textblob']['neutral'],breakdown['textblob']['negative']]
                print("  FinBERT :", _fmt(fb))
                print("  RoBERTa :", _fmt(rb))
                print("  VADER   :", _fmt(vd))
                print("  TextBlob:", _fmt(tb))
                print("  Alpha   :", _fmt(alpha_vec), f"[raw={alpha_label} {alpha_score}]")

            print(f"  --> Ensemble = {label.upper()} (conf={conf:.3f})  "
                  f"[pos={breakdown['combined']['positive']:.3f} "
                  f"neu={breakdown['combined']['neutral']:.3f} "
                  f"neg={breakdown['combined']['negative']:.3f}]")

            rows.append({
                "time_published": item.get("time_published"),
                "source": item.get("source"),
                "url": url,
                "title": headline,
                "pred_label": label,
                "pred_conf": round(conf, 3),
                "alpha_label": alpha_label,
                "alpha_score": alpha_score,
                "body_chars": len(body or ""),
            })

        # Respect Alpha Vantage free-tier rate limits between pages
        if p < pages:
            time.sleep(ALPHAVANTAGE_RATE_SLEEP)
    return rows

def save_results_csv(rows, path=SAVE_CSV_PATH):
    if not path or not rows: return
    cols = ["time_published","source","url","title","pred_label","pred_conf","alpha_label","alpha_score","body_chars"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in cols})
    print(f"\nSaved: {path}")

# -------------- Run --------------
if __name__ == "__main__":
    print(f"Using weights: {WEIGHTS_BASE}  |  NEUTRAL_CAP(base)={NEUTRAL_CAP}  |  SOFTEN_TAU={SOFTEN_TAU} | RB_HEAD_W={RB_HEAD_W} | DEVICE={DEVICE}")
    batch = analyze_alpha_news_batch()
    print(f"\nAnalyzed {len(batch)} Alpha Vantage stories for tickers={TICKERS}")
    if SAVE_CSV_PATH:
        save_results_csv(batch, SAVE_CSV_PATH)
