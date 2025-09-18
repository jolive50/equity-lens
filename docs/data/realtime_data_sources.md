# Real-Time & Streaming Data Strategy

## 1. Market Data Streams
- **Providers:** IEX Cloud (free tier SSE), Yahoo Finance unofficial WebSocket proxies (community maintained, no SLA), Alpha Vantage (free intraday API with 5 calls/min).
- **Transport:** WebSocket streams piped into Kafka topics (`market_quotes`, `market_trades`) and REST polling fallbacks when sockets are unavailable.
- **Processing:** Lightweight Python consumers (asyncio) under `pipelines/realtime/market_consumer.py` (to be developed) push normalized ticks into the online feature store (`data/realtime/feature_store/`).
- **Usage Limits & Compliance:** IEX Cloud free tier allows 50,000 core messages/month and requires attribution when data is displayed; Alpha Vantage limits to 5 API calls per minute and 500 per day; Yahoo Finance community feeds are unofficial and must respect robots.txt plus include "Data from Yahoo Finance" attribution in downstream products.

## 2. News & Social Sentiment Streams
- **Providers:** GDELT Project Event Database (free JSON/CSV feeds updated every 15 minutes), Mediastack free tier (500 requests/month) for headlines, Reddit RSS feeds for specific subreddits, and Pushshift Reddit dumps for historical context.
- **Transport:** Event ingestion through managed Pub/Sub or Kafka Connect connectors. Apply rate-limiting and caching to remain within free quotas.
- **Processing:** Text normalization, language detection, and rapid sentiment scoring using lightweight transformers (DistilBERT) deployed as microservices.
- **Usage Limits & Compliance:** GDELT data is open and requires source acknowledgement; Mediastack free tier enforces 500 requests/month and demands attribution via "Powered by mediastack"; Reddit RSS access should follow Reddit API terms and subreddit-specific rules, with caching to minimize polling load.

## 3. Economic & Corporate Events
- **Providers:** St. Louis Fed FRED API (free with API key, 120 requests/minute), U.S. Securities and Exchange Commission (SEC) RSS feeds for filings, European Central Bank Statistical Data Warehouse (open license API), and Investing.com economic calendar RSS (requires attribution).
- **Transport:** REST polling jobs on cron for providers lacking push interfaces; convert to events stored in `data/realtime/landing/events/`.
- **Processing:** Schema harmonization + event deduping pipelines with idempotent upserts into feature store tables.
- **Usage Limits & Compliance:** FRED requires API key registration and citation of FRED as the source; SEC RSS feeds are public domain but should respect polite polling intervals (≥10 seconds); ECB SDW enforces fair-use via API key and attribution; Investing.com RSS mandates visible attribution when data is displayed.

## 4. Observability & Backfill
- Persist raw streamed payloads to cold storage (`data/realtime/landing/`) for replay/backfill scenarios.
- Use Kafka tiered storage or cloud object storage for retention; align with governance policies.
- Implement checkpointing and lag monitoring dashboards (Prometheus + Grafana) to ensure stream health.
- Log quota usage per provider to alert when approaching free-tier limits and trigger manual review.

## 5. Next Steps
1. Define SLAs and throughput expectations for each stream aligned with free-tier quotas.
2. Prototype a minimal market data WebSocket client and sentiment listener respecting rate limits.
3. Integrate stream ingestion with centralized secrets management, observability stack, and attribution tracking.
