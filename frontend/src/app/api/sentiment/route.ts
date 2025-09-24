import { NextResponse } from "next/server";
import path from "node:path";
import fs from "node:fs/promises";

async function listDirSafe(dir: string): Promise<string[]> { try { return await fs.readdir(dir); } catch { return []; } }
async function statSafe(p: string) { try { return await fs.stat(p); } catch { return null; } }

function resolveRootRelative(...parts: string[]) {
  return path.resolve(process.cwd(), "..", ...parts);
}

async function findLatestJsonl(dir: string): Promise<string | null> {
  const files = await listDirSafe(dir);
  const js = await Promise.all(
    files.filter((f) => f.toLowerCase().endsWith(".jsonl")).map(async (f) => {
      const full = path.join(dir, f);
      const s = await statSafe(full);
      return s ? { full, mtime: s.mtimeMs } : null;
    })
  );
  const present = js.filter(Boolean) as { full: string; mtime: number }[];
  if (!present.length) return null;
  present.sort((a, b) => b.mtime - a.mtime);
  return present[0].full;
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const ticker = (searchParams.get("ticker") || "AAPL").toUpperCase();
  const limit = Math.min(Number(searchParams.get("limit") || 200), 2000);

  const dir = resolveRootRelative("data", "realtime", "feature_store", "sentiment", "finbert");
  const latest = await findLatestJsonl(dir);
  if (!latest) {
    return NextResponse.json({ ticker, items: [] });
  }

  try {
    const raw = await fs.readFile(latest, { encoding: "utf-8" });
    const lines = raw.split(/\r?\n/).filter(Boolean);
    const tail = lines.slice(-limit).map((l) => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean);
    return NextResponse.json({ ticker, items: tail });
  } catch (err: any) {
    return NextResponse.json({ ticker, items: [], error: err?.message });
  }
}


