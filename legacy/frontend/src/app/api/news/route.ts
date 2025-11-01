import { NextResponse } from "next/server";
import path from "node:path";
import fs from "node:fs/promises";

function resolveNewsPath(source: string, ticker: string): string {
  // Frontend lives at <root>/frontend; data lives at <root>/data
  return path.resolve(process.cwd(), "..", "data", "realtime", "landing", "news", source, ticker, "news.jsonl");
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const source = (searchParams.get("source") || "yahoo_finance").replace(/[^a-zA-Z0-9_\-]/g, "");
  const ticker = (searchParams.get("ticker") || "AAPL").toUpperCase().replace(/[^A-Z0-9\.\-]/g, "");
  const limit = Math.min(Number(searchParams.get("limit") || 100), 500);

  try {
    const filePath = resolveNewsPath(source, ticker);
    const content = await fs.readFile(filePath, { encoding: "utf-8" });
    const lines = content.split(/\r?\n/).filter(Boolean);
    const tail = lines.slice(-limit);
    const items = tail.map((l) => {
      try { return JSON.parse(l); } catch { return null; }
    }).filter(Boolean);
    return NextResponse.json({ source, ticker, count: items.length, items });
  } catch (err: unknown) {
    let message = "Not found";
    if (err instanceof Error) message = err.message;
    return NextResponse.json({ error: message, source, ticker, items: [] }, { status: 200 });
  }
}


