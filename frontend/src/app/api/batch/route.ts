import { NextResponse } from "next/server";

function getApiBase(): string {
	const env = process.env.NEXT_PUBLIC_API_BASE || process.env.API_BASE_URL;
	return (env && env.replace(/\/$/, "")) || "http://localhost:8000";
}

export async function POST(request: Request) {
	try {
		const body = await request.json();
		const { tickers, user_tier } = body || {};
		if (!Array.isArray(tickers) || tickers.length === 0) {
			return NextResponse.json({ detail: "tickers array is required" }, { status: 400 });
		}
		const api = getApiBase();
		const res = await fetch(`${api}/analyze/batch`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ tickers, user_tier: user_tier || "premium" })
		});
		const data = await res.json().catch(() => ({}));
		return NextResponse.json(data, { status: res.status });
	} catch (err: any) {
		return NextResponse.json({ detail: err?.message || "Batch failed" }, { status: 500 });
	}
}


