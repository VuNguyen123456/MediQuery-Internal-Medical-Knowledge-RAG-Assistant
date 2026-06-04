# Flask RAG Service — Python 3.12 slim
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for PyMuPDF
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY services/rag/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy RAG source code
COPY services/rag/src/ ./src/

# Copy documents folder
COPY documents/ ./documents/

# HuggingFace model cache — pre-download at build time
# so container startup is instant (no download on first query)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 5000

CMD ["python", "src/app.py"]