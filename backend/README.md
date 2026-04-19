# Lucida Backend

FastAPI backend for model training, scoring, model registry, and auth profile.

## Runtime target
- Python `3.11` or `3.12`
- Works locally with a standard virtualenv
- Deployable as a container via Docker or Railway

The ML dependency stack is pinned to versions with broader binary compatibility so the API can run on typical developer workspaces as well as cloud hosts.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

## API docs
- Swagger: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Verify
```bash
python -m pytest -q
python scripts/preflight.py
```

## Deploy
```bash
docker build -t lucida-backend .
docker run --rm -p 8000:8000 --env-file .env lucida-backend
```

Railway is also configured via `railway.toml` to run the same containerized backend.
