"use client";

import { Card, Inset, Badge } from "@radix-ui/themes";
import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import dayjs from "dayjs";
import relativeTime from "dayjs/plugin/relativeTime";

dayjs.extend(relativeTime);

type NewsItem = {
  id?: string;
  title?: string;
  description?: string;
  url?: string;
  publisher?: string;
  published_time?: string;
  published_time_unix?: number;
};

export default function NewsPage() {
  const [items, setItems] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [ticker, setTicker] = useState("AAPL");
  const [source, setSource] = useState("yahoo_finance");

  useEffect(() => {
    let aborted = false;
    setLoading(true);
    fetch(`/api/news?ticker=${encodeURIComponent(ticker)}&source=${encodeURIComponent(source)}&limit=60`)
      .then((r) => r.json())
      .then((data) => { if (!aborted) setItems(data.items || []); })
      .finally(() => { if (!aborted) setLoading(false); });
    return () => { aborted = true; };
  }, [ticker, source]);

  return (
    <div className="grid gap-6">
      <div className="flex items-end gap-3">
        <div>
          <h1 className="text-2xl font-semibold">News</h1>
          <p className="text-zinc-400">Latest headlines and summaries.</p>
        </div>
        <div className="ml-auto flex gap-2">
          <select value={source} onChange={(e) => setSource(e.target.value)} className="bg-black border border-zinc-800 rounded px-2 py-1">
            <option value="yahoo_finance">Yahoo Finance</option>
            <option value="newsapi">NewsAPI</option>
            <option value="finnhub">Finnhub</option>
          </select>
          <input value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} className="bg-black border border-zinc-800 rounded px-2 py-1 w-28" placeholder="AAPL" />
        </div>
      </div>

      {loading ? (
        <div className="text-zinc-400">Loading…</div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {items.map((item, i) => (
            <motion.a key={(item.id || i) + ""} href={item.url} target="_blank" rel="noreferrer" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.02 * i }}>
              <Card>
                <Inset>
                  <div className="p-4">
                    <div className="text-sm text-zinc-400 mb-1 flex items-center gap-2">
                      <Badge color="gray" variant="soft">{source}</Badge>
                      {item.publisher}
                      <span className="ml-auto text-xs text-zinc-500">
                        {item.published_time_unix ? dayjs.unix(item.published_time_unix).fromNow() : item.published_time ? dayjs(item.published_time).fromNow() : null}
                      </span>
                    </div>
                    <div className="text-lg font-medium line-clamp-2">{item.title || (item.url ? new URL(item.url).hostname : "(untitled)")}</div>
                    {item.description && (
                      <div className="mt-2 text-sm text-zinc-400 line-clamp-3">{item.description}</div>
                    )}
                  </div>
                </Inset>
              </Card>
            </motion.a>
          ))}
          {!items.length && (
            <div className="text-zinc-400">No news found for {ticker}.</div>
          )}
        </div>
      )}
    </div>
  );
}


