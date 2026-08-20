# Go AlphaZero Frontend

Production-ready frontend for the Go AlphaZero game built with TypeScript and Vite.

## Development

```bash
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your browser.

The dev server proxies `/api` to `http://localhost:5000` by default.
To point at a different backend, set `VITE_API_URL` in `.env`.

Example:

```bash
cp .env.example .env
```

## Build

```bash
npm run build
```

The production build will be in the `dist` directory.

## Preview Production Build

```bash
npm run preview
```

## Type Checking

```bash
npm run type-check
```

## Tech Stack

- TypeScript
- Vite
- Canvas API for board rendering

## API Integration

The frontend expects a backend server running on `http://localhost:5000` with the following endpoints:

- `POST /api/game/new` - Start a new game
- `POST /api/game/move` - Make a move
