# Arc Prover Production Deployment Guide

Deploy the Jolt-Atlas zkML prover service for production spending proof generation.

## Prerequisites

- Docker + Docker Compose (recommended) OR Rust 1.75+
- 8GB RAM minimum (16GB recommended for concurrent proofs)
- 4+ CPU cores
- 10GB disk space for models and proofs

## Quick Start (Docker)

```bash
# Build the prover image
docker build -t arc-prover .

# Run with default configuration
docker run -d \
  --name arc-prover \
  -p 3001:3001 \
  -v $(pwd)/models:/app/models \
  -e RUST_LOG=info \
  arc-prover

# Check health
curl http://localhost:3001/health
```

## Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  arc-prover:
    build: .
    ports:
      - "3001:3001"
    volumes:
      - ./models:/app/models
      - proof-cache:/app/cache
    environment:
      - RUST_LOG=info
      - PORT=3001
      - MODELS_DIR=/app/models
      - CACHE_DIR=/app/cache
      - MAX_CONCURRENT_PROOFS=4
    deploy:
      resources:
        limits:
          memory: 16G
        reservations:
          memory: 8G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

volumes:
  proof-cache:
```

Run:
```bash
docker-compose up -d
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3001` | HTTP server port |
| `MODELS_DIR` | `./models` | Path to ONNX model files |
| `CACHE_DIR` | `./cache` | Proof cache directory |
| `RUST_LOG` | `info` | Log level (trace, debug, info, warn, error) |
| `MAX_CONCURRENT_PROOFS` | `4` | Maximum parallel proof generations |
| `PROOF_TIMEOUT_MS` | `30000` | Proof generation timeout |

## API Endpoints

### Health Check
```bash
GET /health

# Response
{
  "status": "healthy",
  "version": "1.0.0",
  "prover": "jolt-atlas",
  "uptime_seconds": 3600,
  "proofs_generated": 142,
  "average_proof_time_ms": 2100
}
```

### List Models
```bash
GET /models

# Response
{
  "models": [
    {
      "id": "spending-model",
      "name": "Spending Policy Model",
      "input_shape": [1, 8],
      "output_shape": [1, 3],
      "version": "1.0.0"
    }
  ]
}
```

### Generate Proof
```bash
POST /prove
Content-Type: application/json

{
  "model_id": "spending-model",
  "inputs": [0.05, 1.0, 0.2, 0.5, 0.95, 1.0, 0.5, 0.5],
  "tag": "spending"
}

# Response
{
  "success": true,
  "proof": {
    "proof": "0x...",
    "proof_hash": "0x7a8b3c4d...",
    "metadata": {
      "model_hash": "0xabcdef...",
      "input_hash": "0x123456...",
      "output_hash": "0x789abc...",
      "proof_size": 48500,
      "generation_time": 2100,
      "prover_version": "1.0.0"
    },
    "tag": "spending",
    "timestamp": 1703894400000
  },
  "inference": {
    "output": 1,
    "raw_output": [1, 87, 12],
    "decision": "approve",
    "confidence": 87
  },
  "generation_time_ms": 2100
}
```

### Verify Proof
```bash
POST /verify
Content-Type: application/json

{
  "proof": "0x...",
  "model_hash": "0xabcdef...",
  "input_hash": "0x123456...",
  "expected_output": [1, 87, 12]
}

# Response
{
  "valid": true,
  "verification_time_ms": 45
}
```

## Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: arc-prover
spec:
  replicas: 2
  selector:
    matchLabels:
      app: arc-prover
  template:
    metadata:
      labels:
        app: arc-prover
    spec:
      containers:
      - name: arc-prover
        image: your-registry/arc-prover:latest
        ports:
        - containerPort: 3001
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "4"
        env:
        - name: RUST_LOG
          value: "info"
        - name: MAX_CONCURRENT_PROOFS
          value: "4"
        volumeMounts:
        - name: models
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 3001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3001
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: prover-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: arc-prover
spec:
  selector:
    app: arc-prover
  ports:
  - port: 3001
    targetPort: 3001
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: arc-prover-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: arc-prover
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Performance Tuning

### Memory Optimization
- Proof generation requires ~4GB per concurrent proof
- Set `MAX_CONCURRENT_PROOFS` based on available memory
- Formula: `max_proofs = (total_memory - 2GB) / 4GB`

### CPU Optimization
- Proof generation is CPU-bound
- Allocate at least 1 core per concurrent proof
- Enable SIMD optimizations in Rust build

### Caching
- Enable proof caching for repeated inputs
- Cache invalidates on model update
- Set `CACHE_DIR` to fast storage (SSD recommended)

## Security Considerations

### Network Security
```bash
# Run behind reverse proxy with TLS
# Example nginx configuration:
server {
    listen 443 ssl;
    server_name prover.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:3001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Rate Limiting
- Implement rate limiting at reverse proxy level
- Recommended: 10 proofs/minute per IP
- Consider API key authentication for production

### Model Security
- Store models in secure, read-only volume
- Verify model hashes on startup
- Log all proof requests for audit trail

## Monitoring

### Prometheus Metrics
```
# Exposed at /metrics
arc_prover_proofs_total{status="success"} 142
arc_prover_proofs_total{status="error"} 3
arc_prover_proof_duration_seconds_bucket{le="2.0"} 120
arc_prover_proof_duration_seconds_bucket{le="4.0"} 140
arc_prover_proof_duration_seconds_bucket{le="10.0"} 142
arc_prover_active_proofs 2
arc_prover_queue_depth 0
```

### Health Monitoring
- Set up alerts for `/health` endpoint failures
- Monitor proof generation latency p99
- Track memory usage for OOM prevention

## Troubleshooting

### Common Issues

**Proof generation timeout**
```
Error: Proof generation exceeded timeout (30000ms)
```
Solution: Increase `PROOF_TIMEOUT_MS` or reduce `MAX_CONCURRENT_PROOFS`

**Out of memory**
```
Error: Cannot allocate memory for proof generation
```
Solution: Reduce `MAX_CONCURRENT_PROOFS` or increase container memory

**Model not found**
```
Error: Model 'spending-model' not found in /app/models
```
Solution: Ensure model file exists at `$MODELS_DIR/spending-model.onnx`

### Debug Mode
```bash
# Run with debug logging
docker run -e RUST_LOG=debug arc-prover

# View real-time logs
docker logs -f arc-prover
```

## Integration with Frontend

Set the prover URL in your Next.js application:

```env
# .env.local
NEXT_PUBLIC_PROVER_URL=https://prover.yourdomain.com
```

The frontend will automatically use this URL for proof generation requests.
