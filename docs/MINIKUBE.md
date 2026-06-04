# Run MediQuery on Minikube (Windows)

## Prerequisites

- Docker Desktop running (`docker version` shows Client **and** Server)
- [Minikube](https://minikube.sigs.k8s.io/docs/start/) installed
- Project root `.env` filled with real API keys
- Pinecone already ingested (`python scripts/ingest.py`)

## Step 1 — Start cluster (once per machine)

```powershell
cd "C:\Users\VuNguyen\OneDrive - George Mason University - O365 Production\Documents\GMU\Internal Medical Knowledge RAG Assistant"

minikube start
```

Optional if Docker context warnings appear:

```powershell
docker context use desktop-linux
```

## Step 2 — Deploy the app

```powershell
.\deploy-minikube.ps1
```

This script will:

1. Enable ingress and metrics-server addons
2. Generate `k8s/secret.yaml` from `.env` (base64, gitignored)
3. Build images inside Minikube's Docker
4. Apply all Kubernetes manifests

First RAG pods may take 1–2 minutes to pass readiness probes.

## Step 3 — Expose ingress (required)

In a **second** PowerShell window (run as Administrator if `localhost` does not respond):

```powershell
minikube tunnel
```

Leave this window open while using the app.

## Step 4 — Verify

```powershell
kubectl get pods -w
```

Wait until `rag`, `api`, and `frontend` pods are `Running` and ready (`1/1`).

Open: **http://localhost**

## Step 5 — Google OAuth

In [Google Cloud Console](https://console.cloud.google.com/) → OAuth client → Authorized redirect URIs, add:

```
http://localhost/auth/callback
```

## Useful commands

```powershell
# RAG logs
kubectl logs -l app=rag -f

# API logs
kubectl logs -l app=api -f

# Restart after code changes
.\deploy-minikube.ps1

# Stop cluster
minikube stop
```

## Troubleshooting

| Problem | Fix |
|--------|-----|
| Script stops on `NativeCommandError` / Docker context stderr | Fixed in `deploy-minikube.ps1` via `Invoke-Native`; warnings do not abort the script |
| Minikube prints `Unable to resolve Docker CLI context default` | Harmless if `minikube start` succeeds; optional: `docker context use desktop-linux` |
| `minikube start failed (exit code 69)` | Docker driver unavailable — finish other minikube/deploy terminals, confirm `docker version` shows Server, then `minikube start` once |
| Deploy froze / restarted in another terminal | Not fatal — wait for one `minikube start` to finish; run `minikube status` before re-running `.\deploy-minikube.ps1` |
| Ingress denied: regex paths with `pathType: Prefix` | Fixed in `k8s/ingress.yaml` — use simple `/api`, `/auth`, `/` Prefix paths; re-apply ingress only if deploy stopped early |
| `auth/success?token=...` shows `Route not found` JSON | Ingress sent `/auth/success` to Express; fixed — only `/auth/login`, `/auth/callback`, etc. go to API; `/auth/success` is React |
| Chat shows `Route not found` / sidebar API unavailable | Frontend called `/api/api/query`; fixed in `api.ts` — rebuild frontend image after pull |
| `ImagePullBackOff` | Re-run `deploy-minikube.ps1` (builds with `imagePullPolicy: Never`) |
| `http://localhost` unreachable | Run `minikube tunnel` in a separate terminal |
| OAuth fails | Redirect URI must be `http://localhost/auth/callback` |
| Script parse errors | Use PowerShell 5.1+; run `.\deploy-minikube.ps1` not `bash` |
