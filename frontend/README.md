# FreshStart Frontend

Next.js frontend for the FreshStart Stock Analysis MVP.

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: CSS Modules / Global CSS

## Getting Started

### Install Dependencies

```bash
npm install
```

### Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

**Note**: Make sure the FastAPI backend is running on `http://localhost:8000`

### Build for Production

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── src/
│   ├── app/
│   │   ├── layout.tsx        # Root layout
│   │   ├── page.tsx          # Main page
│   │   └── globals.css       # Global styles
│   └── components/
│       ├── AnalysisForm.tsx  # Ticker input form
│       └── ResultsDisplay.tsx # Analysis results display
├── public/                    # Static assets
├── package.json
├── tsconfig.json
└── next.config.js
```

## API Integration

API calls target the FastAPI backend directly using an environment variable:

```bash
# frontend/.env.local
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

If the variable is not set the app falls back to `http://localhost:8000`.

## Features

- Single-page stock analysis interface
- Real-time API calls to backend
- Loading states and error handling
- Responsive design
- TypeScript for type safety

## Development

### Adding New Components

Create components in `src/components/`:

```tsx
// src/components/MyComponent.tsx
export default function MyComponent() {
  return <div>Hello World</div>
}
```

### Styling

Global styles are in `src/app/globals.css`. Component-specific styles can be added inline or via CSS modules.

## Team Assignment

**Owner**: Sua (Frontend & Backend)

**Responsibilities**:
- Component development
- API integration
- Styling and UX
- Testing frontend functionality
