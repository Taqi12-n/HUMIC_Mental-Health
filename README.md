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
