# JOLT-Atlas zkML Prover Service
# Multi-stage build for smaller final image

# Build stage
FROM rust:1.75-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy entire workspace
COPY . .

# Build the arc-prover binary
RUN cargo build --release --package arc-prover

# Runtime stage
FROM debian:bookworm-slim

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 -m appuser

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /app/target/release/arc-prover /app/arc-prover

# Copy models
COPY --from=builder /app/arc-prover/models /app/models

# Copy SRS files for proof generation
COPY --from=builder /app/*.srs /app/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV MODELS_DIR=/app/models
ENV PORT=10000
ENV RUST_LOG=info

EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

CMD ["/app/arc-prover"]
