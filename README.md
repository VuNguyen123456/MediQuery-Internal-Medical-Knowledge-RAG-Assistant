# ⬡ MediQuery — Internal Medical Knowledge RAG Assistant

A full-stack Retrieval Augmented Generation (RAG) system that lets authenticated users ask plain-English questions about indexed medical documents and receive accurate, grounded answers with exact source citations — powered by real medical PDFs from NIH, FDA, WHO, and CDC.

---

## What This Is

Medical and clinical teams deal with hundreds of documents — treatment guidelines, drug references, WHO/NIH/FDA publications. Finding specific answers means manually searching through PDFs. This is slow, error-prone, and inefficient.

MediQuery solves this by letting authenticated users ask questions in plain English and instantly receive accurate answers with exact source citations — grounded exclusively in the indexed documents. No hallucinations. No outside knowledge. Every answer is traceable to a specific page in a specific document.

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

**Three microservices, all containerized:**

| Service | Tech | Port | Role |
|---------|------|------|------|
| Frontend | React + TypeScript | 3000 | Chat UI, login, citation display |
| API | Express + TypeScript | 8005 | Auth, JWT validation, proxy to Flask |
| RAG | Flask + Python | 5000 | Document parsing, embeddings, Pinecone search, Gemini generation |

The Flask service is **never directly accessible from the browser** — Express proxies all requests to it internally. API keys (Pinecone, Gemini) never reach the frontend.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | TypeScript + Python | TS for API/frontend, Python for RAG |
| LLM | Gemini 2.0 Flash | Generates grounded answers |
| Vector DB | Pinecone | Stores and searches document embeddings |
| Embedding Model | HuggingFace all-MiniLM-L6-v2 | Converts text to 384-dim vectors (runs locally) |
| Auth | Google OAuth (OIDC) + JWT | Secure enterprise authentication |
| Backend API | Express.js (TypeScript) | User-facing API, auth, proxy layer |
| RAG Service | Flask + LangChain (Python) | Document processing and retrieval |
| Frontend | React + TypeScript | Chat UI with citations |
| Containerization | Docker + Docker Compose | Wires all 3 services together |

---

## How RAG Works

### Document Ingestion (runs once per document)

```
Medical PDF
    ↓
PyMuPDF extracts text + page numbers
    ↓
LangChain RecursiveCharacterTextSplitter
  → 500-word chunks, 50-word overlap
    ↓
HuggingFace all-MiniLM-L6-v2
  → each chunk → 384-dimensional vector
    ↓
Pinecone upsert
  → { id, vector, metadata: { text, source, page } }
```

### Query Pipeline (runs on every user question)

```
User: "What are the side effects of Metformin?"
    ↓
Embed question → 384-dim vector
    ↓
Pinecone cosine similarity search → top 4 chunks
    ↓
LangChain prompt builder injects chunks as context
    ↓
Gemini: "Answer ONLY from provided documents"
    ↓
{ answer: "...", citations: [{ source, page, excerpt }] }
    ↓
React renders answer bubble + citation cards
```

### Why 500-word chunks with 50-word overlap?
- **500 words** = enough context for meaningful answers
- **50-word overlap** = prevents losing context at chunk boundaries (a sentence starting at the end of chunk 3 and finishing in chunk 4 is preserved)
- **Deterministic chunk IDs** = re-running ingestion is safe, Pinecone upsert overwrites on matching ID, no duplicates

---

## Authentication Flow

```
User visits localhost:3000
    ↓
React checks localStorage for valid JWT
    ├── Valid → /chat
    └── Invalid/missing → /login
                ↓
        Click "Sign in with Google"
                ↓
        Google OAuth consent screen
                ↓
        Express receives auth code
        Exchanges code → Google tokens
        Verifies identity → creates JWT (8hr expiry)
                ↓
        Redirect to React /auth/success?token=<jwt>
                ↓
        React stores JWT in localStorage
        Every future request: Authorization: Bearer <jwt>
        Express validates JWT on every /api/* request
        Invalid/expired → 401 → redirect to login
```

---

## Project Structure

```
mediquery/
├── services/
│   ├── api/                          ← Express TypeScript
│   │   ├── src/
│   │   │   ├── routes/
│   │   │   │   ├── auth.ts           ← Google OAuth routes
│   │   │   │   └── query.ts          ← proxies to Flask
│   │   │   ├── middleware/
│   │   │   │   └── authGuard.ts      ← JWT validation
│   │   │   └── server.ts             ← entry point
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   └── Dockerfile
│   │
│   ├── rag/                          ← Flask Python
│   │   ├── src/
│   │   │   ├── ingestion/
│   │   │   │   ├── parser.py         ← PDF text extraction (PyMuPDF)
│   │   │   │   ├── chunker.py        ← RecursiveCharacterTextSplitter
│   │   │   │   └── uploader.py       ← embed + Pinecone upsert
│   │   │   ├── retrieval/
│   │   │   │   └── search.py         ← vector similarity search
│   │   │   ├── generation/
│   │   │   │   ├── prompt.py         ← RAG prompt builder
│   │   │   │   └── llm.py            ← Gemini API call
│   │   │   └── app.py                ← Flask entry point
│   │   ├── requirements.txt
│   │   └── Dockerfile
│   │
│   └── frontend/                     ← React TypeScript
│       ├── src/
│       │   ├── components/
│       │   │   ├── ChatWindow.tsx     ← message bubbles + input
│       │   │   ├── CitationCard.tsx   ← source display
│       │   │   └── DocSidebar.tsx     ← indexed documents list
│       │   ├── pages/
│       │   │   ├── Login.tsx          ← Google OAuth login
│       │   │   ├── Chat.tsx           ← main chat page
│       │   │   └── AuthSuccess.tsx    ← JWT extraction after OAuth
│       │   ├── hooks/
│       │   │   └── useAuth.ts         ← JWT state management
│       │   ├── api.ts                 ← centralized API calls
│       │   └── App.tsx                ← routing + auth protection
│       ├── package.json
│       └── Dockerfile
│
├── documents/                        ← medical PDFs (gitignored)
│   ├── NIH_Diabetes_Guidelines.pdf
│   ├── FDA_Metformin_Label.pdf
│   └── ...
│
├── scripts/
│   └── ingest.py                     ← run once to index documents
│
├── docker-compose.yml                ← wires all 3 services
├── .env.example                      ← environment variable template
└── .dockerignore
```

---

## Indexed Knowledge Base

| Document | Source | Domain |
|----------|--------|--------|
| Child & Adolescent Immunization Schedule | CDC | Immunization |
| Adult Immunization Schedule | CDC | Immunization |
| Invasive Breast Cancer Guidelines | NCCN | Oncology |
| CDC Hypertension Guidelines | CDC | Cardiovascular |
| Hypertension Management Program Toolkit | CDC | Cardiovascular |
| Zestril (Lisinopril) FDA Label | FDA | Drug Reference |
| Glucophage (Metformin) FDA Label | FDA | Drug Reference |
| WHO Diabetes Treatment Guidelines | WHO | Metabolic |

**535 vectors** across 8 documents indexed in Pinecone.

---

## Prerequisites

- Docker Desktop
- Git
- Pinecone account (free tier) — [app.pinecone.io](https://app.pinecone.io)
- Google Cloud project with OAuth 2.0 credentials — [console.cloud.google.com](https://console.cloud.google.com)
- Gemini API key — [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/mediquery.git
cd mediquery
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Fill in your `.env`:

```env
# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=mediquery
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

# Gemini
GEMINI_API_KEY=your_gemini_api_key

# Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# JWT
JWT_SECRET=your_random_secret_string

# URLs
EXPRESS_PORT=8005
EXPRESS_URL=http://localhost:8005
FRONTEND_URL=http://localhost:3000
REACT_APP_API_URL=http://localhost:8005
```

### 3. Add medical PDFs

Place your PDF documents in the `documents/` folder. Free public sources:
- **NIH**: [nhlbi.nih.gov/health-topics/guidelines](https://www.nhlbi.nih.gov/health-topics/guidelines)
- **FDA Drug Labels**: [accessdata.fda.gov/scripts/cder/daf](https://www.accessdata.fda.gov/scripts/cder/daf/)
- **WHO**: [who.int/publications](https://www.who.int/publications/)
- **CDC**: [cdc.gov/vaccines/hcp/acip-recs](https://www.cdc.gov/vaccines/hcp/acip-recs/)

### 4. Ingest documents into Pinecone

```bash
# Create Python virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install RAG dependencies
pip install -r services/rag/requirements.txt

# Run ingestion pipeline
python scripts/ingest.py
```

Verify vectors appear in your Pinecone dashboard at [app.pinecone.io](https://app.pinecone.io).

### 5. Configure Google OAuth

In [Google Cloud Console](https://console.cloud.google.com):
1. APIs & Services → OAuth consent screen → External
2. APIs & Services → Credentials → Create OAuth 2.0 Client ID → Web Application
3. Add authorized redirect URI: `http://localhost:8005/auth/callback`
4. Add authorized JavaScript origin: `http://localhost:3000`

### 6. Start the application

```bash
docker-compose up --build
```

Visit **[http://localhost:3000](http://localhost:3000)**

> **Note:** First build takes 5-10 minutes (downloads base images + HuggingFace model). Subsequent starts are instant.

---

## Development Setup (without Docker)

Run each service manually in separate terminals:

**Terminal 1 — Flask RAG Service:**
```bash
cd services/rag/src
python app.py
# Running on http://localhost:5000
```

**Terminal 2 — Express API:**
```bash
cd services/api
npm install
npm run dev
# Running on http://localhost:8005
```

**Terminal 3 — React Frontend:**
```bash
cd services/frontend
npm install
npm start
# Running on http://localhost:3000
```

---

## API Reference

### Public Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/auth/login` | Redirect to Google OAuth |
| GET | `/auth/callback` | OAuth callback, issues JWT |
| POST | `/auth/logout` | Logout |
| GET | `/auth/me` | Get current user from JWT |

### Protected Endpoints (JWT required)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/query` | Run RAG pipeline |
| GET | `/api/documents` | List indexed documents |

**POST `/api/query` request:**
```json
{ "question": "What are the side effects of Metformin?" }
```

**POST `/api/query` response:**
```json
{
  "answer": "Metformin commonly causes gastrointestinal side effects including nausea, vomiting, and diarrhea, especially during initiation of therapy...",
  "citations": [
    {
      "source": "Metformin.pdf",
      "page": 7,
      "excerpt": "Common adverse effects include nausea, vomiting, diarrhea...",
      "score": 0.91
    }
  ]
}
```

---

## Re-ingesting Documents

The ingestion pipeline is fully idempotent. Chunk IDs are deterministic (based on filename + page + position), so Pinecone upsert overwrites on matching ID — no duplicates.

```bash
# Add new PDFs to documents/ then:
python scripts/ingest.py

# Or ingest a specific file:
python scripts/ingest.py documents/new_document.pdf
```

---

## Security Design

| Mechanism | Purpose |
|-----------|---------|
| Google OAuth (OIDC) | Only authenticated users access the system |
| JWT tokens (8hr expiry) | Stateless session management after login |
| Express backend proxy | Gemini + Pinecone keys never reach browser |
| Flask internal only | Never directly accessible outside Docker network |
| Environment variables | All secrets in `.env`, never hardcoded |
| Docker network isolation | Services communicate internally only |

---

## Example Questions to Ask

**Drug-specific (tests exact label retrieval):**
- "What is the maximum daily dose of Metformin XR?"
- "What are the contraindications for Lisinopril?"
- "Does Metformin require dose adjustment for elderly patients?"

**Cross-document (tests multi-source retrieval):**
- "Can a hypertensive diabetic patient take both Lisinopril and Metformin?"
- "What do the hypertension guidelines say about blood pressure targets?"

**Protocol-specific (tests precise chunk retrieval):**
- "What are the 10 components of the Hypertension Management Program?"
- "What vaccines does the CDC recommend for adults over 65?"


## License

MIT