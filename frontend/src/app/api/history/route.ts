import { NextResponse } from "next/server";

function getApiBase(): string {
	const env = process.env.NEXT_PUBLIC_BACKEND_API_BASE || process.env.BACKEND_API_BASE;
	return (env && env.replace(/\/$/, "")) || "http://localhost:8000";
}

export async function GET(request: Request) {
	const { searchParams } = new URL(request.url);
	const user_id = (searchParams.get("user_id") || "demo").trim();
	const limit = Math.min(Number(searchParams.get("limit") || 100), 1000);
	const api = getApiBase();
	const res = await fetch(`${api}/history?user_id=${encodeURIComponent(user_id)}&limit=${limit}`);
	const data = await res.json().catch(() => ({}));
	return NextResponse.json(data, { status: res.status });
}
