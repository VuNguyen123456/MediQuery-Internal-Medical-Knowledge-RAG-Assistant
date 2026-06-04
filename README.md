# MediQuery — Internal Medical Knowledge RAG Assistant

A full-stack Retrieval Augmented Generation (RAG) system that lets authenticated users ask plain-English questions about indexed medical documents and receive grounded answers with exact source citations — powered by real medical PDFs from NIH, FDA, WHO, and CDC.

---

## What This Is

Medical and clinical teams deal with hundreds of documents — treatment guidelines, drug references, WHO/NIH/FDA publications. Finding specific answers means manually searching through PDFs. This is slow, error-prone, and inefficient.

MediQuery solves this by letting authenticated users ask questions in plain English and instantly receive accurate answers with exact source citations — grounded exclusively in the indexed documents. Every answer is traceable to a specific page in a specific document.

---

## Architecture

```
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│    REACT     │  HTTP   │   EXPRESS    │  HTTP   │    FLASK     │
│  FRONTEND    │ ──────► │     API      │ ──────► │  RAG SERVICE │
│  Port 3000   │         │  Port 8005   │         │  Port 5000   │
└──────────────┘         └──────────────┘         └──────────────┘
                                │                        │
                         Google OAuth            ┌───────┴────────┐
                         JWT validation          │                │
                         API key proxy      Pinecone DB      Gemini API
                                           (vectors)         (answers)
                                                       │
                                               Hugging Face
                                               (embeddings, local)
```

| Service | Tech | Port (local dev) | Role |
|---------|------|------------------|------|
| Frontend | React + TypeScript | 3000 | Chat UI, login, citation display |
| API | Express + TypeScript | 8005 | Auth, JWT validation, proxy to Flask |
| RAG | Flask + Python | 5000 | Embeddings, Pinecone search, Gemini generation |

The Flask service is **never directly accessible from the browser**. Express proxies all requests to it. API keys (Pinecone, Gemini) never reach the frontend.

### Minikube (single entry point)

With Kubernetes ingress, the browser uses **one host** — `http://localhost` — instead of separate ports:

| Path | Backend |
|------|---------|
| `/` | React (nginx) |
| `/api/*` | Express (RAG queries, documents list) |
| `/auth/login`, `/auth/callback`, `/auth/logout`, `/auth/me` | Express (OAuth) |
| `/auth/success` | React (stores JWT after OAuth) |
| `/health` | Express |

Requires `minikube tunnel` in a separate terminal while using the app.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | TypeScript + Python | TS for API/frontend, Python for RAG |
| LLM | Gemini (`gemini-2.5-flash-lite` default) | Grounded answer generation |
| Vector DB | Pinecone | Stores and searches document embeddings |
| Embedding model | HuggingFace `all-MiniLM-L6-v2` | 384-dim vectors (local in RAG service) |
| Auth | Google OAuth + JWT (8h) | Authenticated access |
| API | Express.js (TypeScript) | Auth, JWT, Flask proxy |
| RAG | Flask (Python) | Ingestion, retrieval, prompts |
| Frontend | React + TypeScript | Chat UI with citations |
| Containers | Docker Compose | Local three-service stack |
| Orchestration | Kubernetes (Minikube) | Production-like local deploy |

---

## How RAG Works

### Document ingestion (run once per document batch)

```
Medical PDF
    → PyMuPDF (text + page numbers)
    → LangChain RecursiveCharacterTextSplitter (500 words, 50 overlap)
    → HuggingFace all-MiniLM-L6-v2
    → Pinecone upsert { id, vector, metadata: { text, source, page } }
```

### Query pipeline (every user question)

```
Question → embed → Pinecone top-4 chunks → Gemini (document-only prompt)
    → { answer, citations[] } → React chat + citation cards
```

Ingestion is **idempotent** — deterministic chunk IDs; re-running `scripts/ingest.py` overwrites matching vectors.

---

## Project Structure

```
Internal Medical Knowledge RAG Assistant/
├── services/
│   ├── api/                 # Express — auth, JWT, Flask proxy
│   │   └── src/routes/auth.ts, query.ts
│   ├── rag/                 # Flask — ingestion, retrieval, Gemini
│   │   └── src/app.py, generation/prompt.py, retrieval/search.py
│   └── frontend/            # React — chat, login, sidebar
│       └── src/api.ts, components/, pages/
├── documents/               # Medical PDFs (tracked in git; add your own)
├── scripts/
│   ├── ingest.py            # Index PDFs into Pinecone
│   └── generate-k8s-secret.ps1  # Build k8s/secret.yaml from .env (Windows)
├── k8s/                     # Kubernetes manifests
│   ├── *-deployment.yaml, ingress.yaml, configmap.yaml
│   └── secret.yaml.example  # Template — real secret.yaml is gitignored
├── docker-compose.yml
├── deploy-minikube.ps1      # Windows Minikube deploy script
├── deploy.sh                # Bash deploy (Docker + kubectl)
├── docs/MINIKUBE.md         # Detailed Minikube guide
└── .env                     # Secrets (gitignored — create locally)
```

---

## Indexed Knowledge Base

Eight public medical PDFs are included (immunization schedules, hypertension toolkit, Lisinopril/Metformin labels, WHO diabetes guidelines, etc.). After ingestion, expect on the order of **500+ vectors** in Pinecone.

---

## Prerequisites

- **Docker Desktop** (for Compose or Minikube docker driver)
- **Git**
- **Python 3.12+** (for ingestion and native Flask dev)
- **Node.js 20+** (for native API/frontend dev)
- [Pinecone](https://app.pinecone.io) account and API key
- [Google AI Studio](https://aistudio.google.com/apikey) API key for Gemini (`Generative Language API`)
- [Google Cloud](https://console.cloud.google.com) OAuth 2.0 client (Web application)

Optional for Kubernetes local deploy:

- [Minikube](https://minikube.sigs.k8s.io/docs/start/)
- `kubectl`

---

## Environment Variables

Create a `.env` file in the project root (never commit it). Example:

```env
# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=mediquery
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Gemini (use an AI Studio key — not Agent Platform-only keys)
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash-lite

# Google OAuth
GOOGLE_CLIENT_ID=your_client_id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your_client_secret

# JWT
JWT_SECRET=your_long_random_secret

# --- Docker Compose / native dev (separate ports) ---
EXPRESS_PORT=8005
EXPRESS_URL=http://localhost:8005
FRONTEND_URL=http://localhost:3000
REACT_APP_API_URL=http://localhost:8005

# --- Minikube ingress (single host) — use when deploying to K8s ---
# EXPRESS_URL=http://localhost
# FRONTEND_URL=http://localhost
# REACT_APP_API_URL=http://localhost/api
```

---

## Quick Start (Docker Compose)

### 1. Clone and configure

```bash
git clone https://github.com/VuNguyen123456/MediQuery-Internal-Medical-Knowledge-RAG-Assistant.git
cd MediQuery-Internal-Medical-Knowledge-RAG-Assistant
```

Create `.env` as above (Compose / native URLs).

### 2. Ingest documents into Pinecone

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1   # macOS/Linux: source venv/bin/activate
pip install -r services/rag/requirements.txt
python scripts/ingest.py
```

Confirm vectors in the [Pinecone console](https://app.pinecone.io).

### 3. Google OAuth (Compose / native)

In Google Cloud → Credentials → your OAuth client:

| Setting | Value |
|---------|--------|
| Authorized redirect URIs | `http://localhost:8005/auth/callback` |
| Authorized JavaScript origins | `http://localhost:3000` |

If the app is in **Testing**, add your Google account under **Test users**.

### 4. Start services

```bash
docker compose up --build
```

Open **http://localhost:3000**.

First build can take several minutes (images + embedding model cache in the RAG image).

---

## Development Setup (no Docker)

Three terminals from the project root (with `.env` loaded and ingestion done):

```powershell
# Terminal 1 — Flask
cd services\rag\src
python app.py

# Terminal 2 — Express
cd services\api
npm install
npm run dev

# Terminal 3 — React
cd services\frontend
npm install
npm start
```

Use **http://localhost:3000** and OAuth redirect `http://localhost:8005/auth/callback`.

---

## Minikube (Kubernetes local)

Production-style deploy: ingress on **http://localhost**, internal service DNS, secrets via Kubernetes.

**Detailed steps:** [docs/MINIKUBE.md](docs/MINIKUBE.md)

**Short version (Windows PowerShell):**

```powershell
# Prerequisites: Docker Desktop running, minikube, kubectl
minikube start
minikube addons enable ingress
minikube addons enable metrics-server

# .env at project root; then:
.\deploy-minikube.ps1

# Second terminal — keep open:
minikube tunnel
```

**Google OAuth for Minikube:**

| Setting | Value |
|---------|--------|
| Authorized redirect URIs | `http://localhost/auth/callback` |
| Authorized JavaScript origins | `http://localhost` (recommended) |

Generate `k8s/secret.yaml` from `.env` (gitignored):

```powershell
.\scripts\generate-k8s-secret.ps1
```

**Rebuild frontend only** (after UI/API URL changes):

```powershell
minikube -p minikube docker-env --shell powershell | Invoke-Expression
docker build -t mediquery-frontend:latest --build-arg REACT_APP_API_URL=http://localhost/api -f services/frontend/Dockerfile .
kubectl rollout restart deployment frontend-deployment
```

**Rebuild RAG only** (after `app.py` / documents path changes):

```powershell
minikube -p minikube docker-env --shell powershell | Invoke-Expression
docker build -t mediquery-rag:latest -f services/rag/Dockerfile .
kubectl rollout restart deployment rag-deployment
```

---

## Authentication Flow

```
User → /login → Sign in with Google → /auth/login (Express)
    → Google consent → /auth/callback (Express) → JWT
    → /auth/success (React stores token) → /chat
```

All `/api/*` requests send `Authorization: Bearer <jwt>`. Expired or invalid tokens return 401 and redirect to login.

---

## API Reference

### Public (Express)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/auth/login` | Start Google OAuth |
| GET | `/auth/callback` | OAuth callback; issues JWT |
| POST | `/auth/logout` | Logout |
| GET | `/auth/me` | Current user from JWT |

### Protected (JWT required)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/query` | Run RAG pipeline |
| GET | `/api/documents` | List PDFs in knowledge base |

**POST `/api/query`**

```json
{ "question": "What are the side effects of Metformin?" }
```

**Response**

```json
{
  "answer": "...",
  "citations": [
    { "source": "Metformin.pdf", "page": 7, "excerpt": "...", "score": 0.91 }
  ]
}
```

---

## Re-ingesting Documents

```bash
python scripts/ingest.py
# Or one file:
python scripts/ingest.py documents/your_new_file.pdf
```

---

## Security

| Mechanism | Purpose |
|-----------|---------|
| Google OAuth | Authenticated users only |
| JWT (8h) | Stateless sessions |
| Express proxy | External API keys stay server-side |
| Flask internal / ClusterIP | RAG not exposed to browser |
| `.env` + `k8s/secret.yaml` gitignored | Secrets not in repository |

---

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| `redirect_uri_mismatch` | OAuth redirect URI must match `EXPRESS_URL` + `/auth/callback` (8005 for Compose, `http://localhost` for Minikube) |
| `missing_code` / `Route not found` on `/auth/success` | Apply latest `k8s/ingress.yaml`; `/auth/success` must route to React |
| Chat `Route not found` on Minikube | Rebuild frontend after `api.ts` fix; use `REACT_APP_API_URL=http://localhost/api` |
| Sidebar "No documents indexed" but chat works | Rebuild RAG image (`/documents` path in container) |
| Gemini 403 / quota | Use [AI Studio](https://aistudio.google.com/apikey) key; set `GEMINI_MODEL=gemini-2.5-flash-lite` |
| Minikube / Docker issues | See [docs/MINIKUBE.md](docs/MINIKUBE.md) |

---

## Example Questions

- "What are the contraindications for Lisinopril in patients with renal artery stenosis?"
- "What blood pressure threshold triggers an EHR alert in the HMP toolkit?"
- "What is the maximum daily dose of Metformin XR according to the FDA label?"
- "What vaccines does the CDC recommend for adults over 65?"

---

## License

MIT
