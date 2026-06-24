# MindVoice AI

Repository structure:

```text
/
├── backend/
├── frontend/
└── run.bat
```

## Development

Run `run.bat` from the repository root to start both services.

If you want to start them manually:

```bash
cd frontend
npm install
npm run dev
```

```bash
cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

## Production storage

Audio uploads and analysis results are persisted in PostgreSQL when `DATABASE_URL`
is set. For Vercel deployment, create a PostgreSQL database such as Neon,
Supabase, or Vercel Postgres, then add its connection string as the
`DATABASE_URL` environment variable in the Vercel project settings.

Without `DATABASE_URL`, the backend uses a local SQLite fallback for development.
That fallback is not persistent on serverless deployments.
