// equity-lens/frontend/src/app/api/analyze/route.ts
// Next.js API route that either calls FastAPI or returns mock data
// FastAPI가 켜져 있으면 그쪽으로 프록시, 아니면 더미 데이터 리턴

import { NextRequest, NextResponse } from 'next/server'

export async function POST(req: NextRequest) {
  // 프론트에서 보낸 JSON 파싱 / parse incoming JSON
  const body = await req.json()
  const ticker = (body?.ticker || '').toUpperCase()

  // 1) try to call local FastAPI first 먼저 로컬에서 도는 FastAPI로 보내보기
  try {
    const fastapiRes = await fetch('http://127.0.0.1:8000/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker }),
    })

    if (fastapiRes.ok) {
      const data = await fastapiRes.json()
      return NextResponse.json(data, { status: 200 })
    }

  } catch (err) {

  }

  // 2) if FastAPI is not running, return mock data so UI still works (FastAPI가 안 떠 있을 때는 임시 데이터 주기)
  return NextResponse.json(
    {
      ticker,
      prediction: {
        direction: 'up',
        confidence: 0.7,
      },
      sentiment: {
        label: 'positive',
        score: 0.6,
        model: 'mock-finbert',
      },
      final_decision: 'BUY',
      explanation:
        'FastAPI backend is not running. This is mock data from Next.js route. 실제 백엔드가 뜨면 이 메시지는 사라집니다.',
    },
    { status: 200 },
  )
}
