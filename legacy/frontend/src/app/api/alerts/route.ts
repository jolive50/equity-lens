import { NextResponse } from "next/server";

function getApiBase(): string {
	const env = process.env.NEXT_PUBLIC_BACKEND_API_BASE || process.env.BACKEND_API_BASE;
	return (env && env.replace(/\/$/, "")) || "http://localhost:8000";
}

export async function GET(request: Request) {
	const { searchParams } = new URL(request.url);
	const user_id = (searchParams.get("user_id") || "demo").trim();
	const api = getApiBase();
	const res = await fetch(`${api}/alerts?user_id=${encodeURIComponent(user_id)}`);
	const data = await res.json().catch(() => ({}));
	return NextResponse.json(data, { status: res.status });
}

export async function POST(request: Request) {
	const { searchParams } = new URL(request.url);
	const user_id = (searchParams.get("user_id") || "demo").trim();
	const body = await request.json().catch(() => ({}));
	const ticker = (body?.ticker || "").toString();
	const condition = (body?.condition || "").toString();
	const threshold = Number(body?.threshold);
	if (!ticker || !condition || Number.isNaN(threshold)) {
		return NextResponse.json({ detail: "ticker, condition, threshold are required" }, { status: 400 });
	}
	const api = getApiBase();
	const url = `${api}/alerts?user_id=${encodeURIComponent(user_id)}&ticker=${encodeURIComponent(ticker)}&condition=${encodeURIComponent(condition)}&threshold=${encodeURIComponent(String(threshold))}`;
	const res = await fetch(url, { method: "POST" });
	const data = await res.json().catch(() => ({}));
	return NextResponse.json(data, { status: res.status });
}

export async function DELETE(request: Request) {
	const { searchParams } = new URL(request.url);
	const user_id = (searchParams.get("user_id") || "demo").trim();
	const ticker = (searchParams.get("ticker") || "").trim();
	const condition = (searchParams.get("condition") || "").trim();
	if (!ticker) return NextResponse.json({ detail: "ticker is required" }, { status: 400 });
	const api = getApiBase();
	const url = `${api}/alerts?user_id=${encodeURIComponent(user_id)}&ticker=${encodeURIComponent(ticker)}${condition ? `&condition=${encodeURIComponent(condition)}` : ""}`;
	const res = await fetch(url, { method: "DELETE" });
	const data = await res.json().catch(() => ({}));
	return NextResponse.json(data, { status: res.status });
}
