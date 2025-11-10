import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'FreshStart Stock Analysis',
  description: 'AI-powered stock predictions with sentiment analysis',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
