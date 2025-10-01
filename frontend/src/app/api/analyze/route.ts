import { NextResponse } from "next/server";

function getApiBase(): string {
	const env = process.env.NEXT_PUBLIC_BACKEND_API_BASE || process.env.BACKEND_API_BASE;
	return (env && env.replace(/\/$/, "")) || "http://localhost:8000";
}

export async function POST(request: Request) {
	try {
		const body = await request.json().catch(() => ({}));
		const { ticker, user_tier } = body || {};

		if (!ticker || typeof ticker !== "string") {
			return NextResponse.json({ detail: "Ticker is required" }, { status: 400 });
		}

		const api = getApiBase();
		const upstream = await fetch(`${api}/analyze`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ ticker: String(ticker).toUpperCase(), user_tier: user_tier || "basic" })
		});
		const data = await upstream.json().catch(() => ({}));
		return NextResponse.json(data, { status: upstream.status });
	} catch (error: any) {
		return NextResponse.json({ detail: error?.message || "Analysis failed" }, { status: 500 });
	}
}
