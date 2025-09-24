import { NextResponse } from "next/server";
import path from "node:path";
import fs from "node:fs/promises";

async function listDirSafe(dir: string): Promise<string[]> {
  try { return await fs.readdir(dir); } catch { return []; }
}

async function statSafe(p: string) {
  try { return await fs.stat(p); } catch { return null; }
}

function resolveRootRelative(...parts: string[]) {
  // frontend lives at <root>/frontend
  return path.resolve(process.cwd(), "..", ...parts);
}

async function findLatestCsv(dir: string): Promise<string | null> {
  const files = await listDirSafe(dir);
  const csvs = await Promise.all(
    files.filter((f) => f.toLowerCase().endsWith(".csv")).map(async (f) => {
      const full = path.join(dir, f);
      const s = await statSafe(full);
      return s ? { full, mtime: s.mtimeMs } : null;
    })
  );
  const present = csvs.filter(Boolean) as { full: string; mtime: number }[];
  if (!present.length) return null;
  present.sort((a, b) => b.mtime - a.mtime);
  return present[0].full;
}

function parseCsv(content: string): { headers: string[]; rows: string[][] } {
  const lines = content.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (!lines.length) return { headers: [], rows: [] };
  const headers = lines[0].split(",");
  const rows = lines.slice(1).map((l) => l.split(","));
  return { headers, rows };
}

function extractDateClose(headers: string[], rows: string[][], variant: "generic" | "free_apple" = "generic") {
  const headerIndex = (name: string) => headers.findIndex((h) => h.toLowerCase() === name.toLowerCase());
  let dateIdx = headerIndex("date");
  let closeIdx = headerIndex("adj_close");
  if (closeIdx < 0) closeIdx = headerIndex("adjusted close");
  if (closeIdx < 0) closeIdx = headerIndex("close");

  if (variant === "free_apple") {
    // finance-charts-apple.csv has date at col 0 and close at col 4
    dateIdx = 0; closeIdx = 4;
  }

  const points = rows.map((r) => {
    const d = r[dateIdx];
    const c = parseFloat(r[closeIdx]);
    return (isFinite(c) && d) ? { date: d, close: c } : null;
  }).filter(Boolean) as { date: string; close: number }[];

  return points;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const ticker = (searchParams.get("ticker") || "AAPL").toUpperCase().replace(/[^A-Z0-9\.\-]/g, "");
  const source = (searchParams.get("source") || "alpha_vantage").replace(/[^a-zA-Z0-9_\-]/g, "");
  const limit = Math.min(Number(searchParams.get("limit") || 200), 2000);

  try {
    let filePath: string | null = null;
    let variant: "generic" | "free_apple" = "generic";
    let sourceUsed = source;

    if (source === "alpha_vantage") {
      filePath = await findLatestCsv(resolveRootRelative("data", "batch", "raw", "market", "alpha_vantage", ticker));
    } else if (source === "tiingo") {
      filePath = await findLatestCsv(resolveRootRelative("data", "batch", "raw", "market", "tiingo", ticker));
    } else if (source === "yahoo_finance_free") {
      // Single CSV with Apple history
      const p = resolveRootRelative("data", "batch", "raw", "market", "yahoo_finance", "free_download", "finance-charts-apple.csv");
      const s = await statSafe(p);
      if (s && ticker === "AAPL") { filePath = p; variant = "free_apple"; }
    }

    // Fallback: if no provider file found, try Yahoo free CSV for AAPL
    if (!filePath && ticker === "AAPL") {
      const p = resolveRootRelative("data", "batch", "raw", "market", "yahoo_finance", "free_download", "finance-charts-apple.csv");
      const s = await statSafe(p);
      if (s) { filePath = p; variant = "free_apple"; sourceUsed = "yahoo_finance_free"; }
    }

    if (!filePath) {
      return NextResponse.json({ ticker, source: sourceUsed, items: [] });
    }

    const raw = await fs.readFile(filePath, { encoding: "utf-8" });
    const { headers, rows } = parseCsv(raw);
    const points = extractDateClose(headers, rows, variant);
    const tail = points.slice(-limit);
    return NextResponse.json({ ticker, source: sourceUsed, count: tail.length, items: tail });
  } catch (err: any) {
    return NextResponse.json({ ticker, source, error: err?.message || "Failed" }, { status: 200 });
  }
}


