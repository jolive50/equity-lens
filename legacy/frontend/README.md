# StockSense Frontend

This package hosts the StockSense web experience built on the Next.js App Router.
It consumes the FastAPI backend (`pipelines.realtime.api`) to surface multi-agent
stock analysis, watchlists, and reporting.

## Prerequisites

- Node.js 18 or newer (bundled npm works fine)
- Backend API running locally at `http://localhost:8000`

## Installation

From the project root:

```bash
cd frontend
npm install
```

Create a local environment file whenever you need to override defaults:

```bash
cp .env.local.example .env.local   # PowerShell: Copy-Item .env.local.example .env.local
```

`NEXT_PUBLIC_BACKEND_API_BASE` should match the address of the FastAPI service.

## Development

```bash
npm run dev          # Start the dev server on http://localhost:3000
npm run lint         # ESLint checks (uses next lint)
npm test             # Jest + React Testing Library
npm run build        # Production build preview
```

The App Router lives under `src/app/` with route groups for dashboards, news, and
API proxies (`src/app/api/*`). Reusable UI lives in `src/components/`.

## Linking to the Backend

The API routes under `src/app/api/` act as thin proxies to the Python backend. For
new endpoints:

1. Add a handler in `src/app/api/<endpoint>/route.ts`.
2. Forward requests to `process.env.NEXT_PUBLIC_BACKEND_API_BASE`.
3. Surface typed responses through a shared client (planned: `frontend/lib/api-client.ts`).

## Testing Strategy

- **Unit/component tests** - `npm test` executes Jest suites covering UI components.
- **Integration** - use Playwright or Cypress (not yet configured) for end-to-end flows.
- **Backend parity** - run `python scripts/run_tests.py` in the repo root to ensure
  the API contract matches the expectations in the proxy routes.

## Project Layout (abridged)

```
frontend/
|-- public/                    # Static assets
|-- src/
|   |-- app/
|   |   |-- api/               # Proxy routes for backend endpoints
|   |   |-- dashboard/         # Authenticated dashboard shell
|   |   |-- news/              # Market news page
|   |   |-- layout.tsx         # Global page layout
|   |   `-- page.tsx           # Landing page defaults
|   |-- components/            # Charts, loaders, error boundaries
|   `-- styles/                # Global styles if needed
|-- package.json
`-- README.md
```

## Troubleshooting

- If API calls fail, verify the FastAPI server is running and the `.env.local`
  value for `NEXT_PUBLIC_BACKEND_API_BASE` is correct.
- Next.js caches responses aggressively; restart `npm run dev` after updating
  environment variables.
- Tailwind, Radix, and chart libraries are already configured-import components
  directly from `src/components/`.
