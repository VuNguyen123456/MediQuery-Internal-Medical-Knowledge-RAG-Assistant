# Deploy MediQuery to local Minikube (Windows PowerShell)
#
# HOW TO RUN:
#   1. Start Docker Desktop and wait until it is running.
#   2. In PowerShell, from this project folder:
#        .\deploy-minikube.ps1
#   3. In a SECOND terminal (Admin recommended):
#        minikube tunnel
#   4. Open http://localhost and add OAuth redirect:
#        http://localhost/auth/callback
#
# Prerequisites: minikube, kubectl, Docker; .env at project root

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root

function Invoke-Native {
    param(
        [string]$Label,
        [scriptblock]$Command,
        [switch]$IgnoreStderr,
        [switch]$CaptureOutput
    )
    $prevEA = $ErrorActionPreference
    $prevNative = $null
    if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
        $prevNative = $PSNativeCommandUseErrorActionPreference
        $PSNativeCommandUseErrorActionPreference = $false
    }
    $ErrorActionPreference = "Continue"
    try {
        if ($CaptureOutput) {
            if ($IgnoreStderr) {
                $out = & $Command 2>$null
            } else {
                $out = & $Command
            }
        } elseif ($IgnoreStderr) {
            & $Command 2>$null
        } else {
            & $Command
        }
        $exitCode = $LASTEXITCODE
        if ($exitCode -ne 0) {
            throw "$Label failed (exit code $exitCode)"
        }
        if ($CaptureOutput) { return $out }
    } finally {
        $ErrorActionPreference = $prevEA
        if ($null -ne $prevNative) {
            $PSNativeCommandUseErrorActionPreference = $prevNative
        }
    }
}

function Test-MinikubeRunning {
    $prevEA = $ErrorActionPreference
    if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
        $PSNativeCommandUseErrorActionPreference = $false
    }
    $ErrorActionPreference = "Continue"
    try {
        $json = minikube status -o json 2>$null
        if ($LASTEXITCODE -ne 0) { return $false }
        $status = $json | ConvertFrom-Json
        return ($status.Host -eq "Running")
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $prevEA
    }
}

Write-Host ""
Write-Host "========================================"
Write-Host "  MediQuery - Minikube Deployment"
Write-Host "========================================"

Write-Host "Note: Docker context warnings from minikube are safe to ignore, or run: docker context use desktop-linux"
Write-Host "      Run only one deploy at a time; do not start minikube in two terminals at once."

# --- Minikube cluster ---
if (Test-MinikubeRunning) {
    Write-Host "[0/6] Minikube cluster already running."
} else {
    Write-Host "[0/6] Starting Minikube..."
    try {
        Invoke-Native -Label "minikube start" -IgnoreStderr { minikube start }
    } catch {
        if (Test-MinikubeRunning) {
            Write-Host "  Cluster is up (start returned an error; continuing deploy)."
        } else {
            Write-Host ""
            Write-Host "minikube start failed. Common fixes:"
            Write-Host "  - Wait for any other minikube start to finish, then retry"
            Write-Host "  - Ensure Docker Desktop is fully running: docker version"
            Write-Host "  - Exit code 69 = Docker driver unavailable: docker context use desktop-linux"
            Write-Host "  - Reset: minikube delete && minikube start"
            throw
        }
    }
}

Write-Host "[0/6] Enabling addons: ingress and metrics-server..."
Invoke-Native -Label "minikube addons enable ingress" -IgnoreStderr {
    minikube addons enable ingress
}
Invoke-Native -Label "minikube addons enable metrics-server" -IgnoreStderr {
    minikube addons enable metrics-server
}

# --- Secrets from .env (base64) ---
Write-Host "[1/6] Generating k8s/secret.yaml from .env..."
& "$Root\scripts\generate-k8s-secret.ps1"

# --- Build images inside Minikube's Docker ---
Write-Host "[2/6] Pointing Docker at Minikube daemon..."
$dockerEnv = Invoke-Native -Label "minikube docker-env" -IgnoreStderr -CaptureOutput {
    minikube -p minikube docker-env --shell powershell
}
Invoke-Expression ($dockerEnv -join "`n")

Write-Host "[3/6] Building Docker images (may take several minutes)..."
Invoke-Native -Label "docker build rag" -IgnoreStderr {
    docker build -t mediquery-rag:latest -f services/rag/Dockerfile .
}
Invoke-Native -Label "docker build api" -IgnoreStderr {
    docker build -t mediquery-api:latest -f services/api/Dockerfile .
}
Invoke-Native -Label "docker build frontend" -IgnoreStderr {
    docker build -t mediquery-frontend:latest `
        --build-arg REACT_APP_API_URL=http://localhost/api `
        -f services/frontend/Dockerfile .
}
Write-Host "  Images built in Minikube Docker"

Write-Host "[4/6] Applying ConfigMap and Secrets..."
Invoke-Native -Label "kubectl apply configmap" { kubectl apply -f k8s/configmap.yaml }
Invoke-Native -Label "kubectl apply secret" { kubectl apply -f k8s/secret.yaml }

Write-Host "[5/6] Deploying services, ingress, HPA..."
Invoke-Native -Label "kubectl apply documents pvc" { kubectl apply -f k8s/documents-pvc.yaml }
Invoke-Native -Label "kubectl apply rag" { kubectl apply -f k8s/rag-deployment.yaml }
Invoke-Native -Label "kubectl apply api" { kubectl apply -f k8s/api-deployment.yaml }
Invoke-Native -Label "kubectl apply frontend" { kubectl apply -f k8s/frontend-deployment.yaml }
Invoke-Native -Label "kubectl apply ingress" { kubectl apply -f k8s/ingress.yaml }
Invoke-Native -Label "kubectl apply hpa" { kubectl apply -f k8s/hpa.yaml }

Write-Host "[6/6] Status"
Invoke-Native -Label "kubectl get pods" { kubectl get pods }
Invoke-Native -Label "kubectl get svc" { kubectl get svc }
Invoke-Native -Label "kubectl get ingress" { kubectl get ingress }

Write-Host ""
Write-Host "========================================"
Write-Host "  Next steps"
Write-Host "========================================"
Write-Host '1. In a NEW terminal, run:  minikube tunnel'
Write-Host '2. Watch pods:             kubectl get pods -w'
Write-Host '3. Open app:               http://localhost'
Write-Host '4. Google OAuth redirect:  http://localhost/auth/callback'
Write-Host ""
Write-Host 'Logs:  kubectl logs -l app=rag -f'
