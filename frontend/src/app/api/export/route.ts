import { NextResponse } from "next/server";

function getApiBase(): string {
	const env = process.env.NEXT_PUBLIC_BACKEND_API_BASE || process.env.BACKEND_API_BASE;
	return (env && env.replace(/\/$/, "")) || "http://localhost:8000";
}

export async function GET(request: Request) {
	const { searchParams } = new URL(request.url);
	const tickers = (searchParams.get("tickers") || "").trim();
	const max_rows = Math.min(Number(searchParams.get("max_rows") || 5000), 5000);
	const user_tier = (searchParams.get("user_tier") || "premium").trim();

	if (!tickers) {
		return new NextResponse("tickers is required", { status: 400 });
	}

	const api = getApiBase();
	const upstream = await fetch(`${api}/export/predictions.csv?tickers=${encodeURIComponent(tickers)}&max_rows=${max_rows}&user_tier=${encodeURIComponent(user_tier)}`);
	const body = upstream.body;
	const headers = new Headers(upstream.headers);
	headers.set("Content-Type", "text/csv");
	headers.set("Content-Disposition", headers.get("Content-Disposition") || "attachment; filename=predictions.csv");
	return new NextResponse(body, { status: upstream.status, headers });
}


