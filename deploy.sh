#!/bin/bash
# deploy.sh — Deploy MediQuery to Kubernetes
#
# USAGE:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# PREREQUISITES:
#   - kubectl configured and pointing at your cluster
#   - Docker images built and pushed to a registry
#     (or built locally for minikube)
#   - k8s/secret.yaml filled in with real base64-encoded values

set -e  # exit on any error

echo ""
echo "========================================"
echo "  MediQuery — Kubernetes Deployment"
echo "========================================"

# ─── Step 1: Build and tag Docker images ─────────────────────────────────────
echo ""
echo "[1/5] Building Docker images..."

# For minikube: MINIKUBE=1 ./deploy.sh
if [ -n "$MINIKUBE" ]; then
  eval $(minikube docker-env)
fi

docker build -t mediquery-rag:latest -f services/rag/Dockerfile .
docker build -t mediquery-api:latest -f services/api/Dockerfile .
docker build -t mediquery-frontend:latest \
  --build-arg REACT_APP_API_URL=http://localhost/api \
  -f services/frontend/Dockerfile .

echo "  ✓ Images built"

# ─── Step 2: Apply ConfigMap and Secrets ─────────────────────────────────────
echo ""
echo "[2/5] Applying ConfigMap and Secrets..."
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml
echo "  ✓ Config and secrets applied"

# ─── Step 3: Deploy services ─────────────────────────────────────────────────
echo ""
echo "[3/5] Deploying services..."
kubectl apply -f k8s/rag-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml
echo "  ✓ Deployments and Services applied"

# ─── Step 4: Apply Ingress ───────────────────────────────────────────────────
echo ""
echo "[4/5] Applying Ingress..."
kubectl apply -f k8s/ingress.yaml
echo "  ✓ Ingress applied"

# ─── Step 5: Apply HPA ───────────────────────────────────────────────────────
echo ""
echo "[5/5] Applying HorizontalPodAutoscalers..."
kubectl apply -f k8s/hpa.yaml
echo "  ✓ HPA applied"

# ─── Status ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Deployment Complete"
echo "========================================"
echo ""
echo "Checking pod status (may take 60s for RAG to become ready)..."
echo ""
kubectl get pods
echo ""
echo "Services:"
kubectl get services
echo ""
echo "Ingress:"
kubectl get ingress
echo ""
echo "HPA status:"
kubectl get hpa
echo ""
echo "  Visit: http://localhost"
echo ""
echo "To watch pods come up:"
echo "  kubectl get pods -w"
echo ""
echo "To view logs:"
echo "  kubectl logs -l app=rag -f"
echo "  kubectl logs -l app=api -f"
echo "  kubectl logs -l app=frontend -f"
