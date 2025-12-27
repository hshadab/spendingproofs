# Spending Proofs Implementation Plan

## Executive Summary

Build a "Policy Proofs as a Service" platform using [jolt-atlas](https://github.com/ICME-Lab/jolt-atlas) as the core proving engine. The system will enable AI agents to generate cryptographic proofs that their decisions followed predefined policies, with verification available across multiple blockchains.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SPENDING PROOFS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Demo UI (React/Next.js)                       │   │
│  │  - Policy selector            - Real-time proof progress              │   │
│  │  - Input sliders/forms        - Multi-chain verification              │   │
│  │  - Proof visualization        - Download proof artifacts              │   │
│  └────────────────────────────────────────┬─────────────────────────────┘   │
│                                           │                                  │
│                                           ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         API Layer (Rust/Axum)                         │   │
│  │                                                                        │   │
│  │  POST /v1/prove     - Generate proof for policy decision              │   │
│  │  POST /v1/verify    - Verify proof locally                            │   │
│  │  GET  /v1/policies  - List available policy templates                 │   │
│  │  GET  /v1/contracts - Get verification contract addresses             │   │
│  │  WS   /v1/prove/stream - Real-time proof generation progress          │   │
│  └────────────────────────────────────────┬─────────────────────────────┘   │
│                                           │                                  │
│                                           ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Core Prover Engine (jolt-atlas)                    │   │
│  │                                                                        │   │
│  │  ┌────────────────┐  ┌──────────────────┐  ┌───────────────────────┐ │   │
│  │  │ Policy Models  │  │  ONNX Tracer     │  │  zkml-jolt-core       │ │   │
│  │  │ (ONNX)        │──▶│  (Execution)     │──▶│  (Proof Generation)  │ │   │
│  │  └────────────────┘  └──────────────────┘  └───────────────────────┘ │   │
│  └────────────────────────────────────────┬─────────────────────────────┘   │
│                                           │                                  │
│                                           ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                 Multichain Verification Layer (Solidity)              │   │
│  │                                                                        │   │
│  │    ┌─────────┐  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌─────────┐   │   │
│  │    │  Base   │  │  Avax   │  │ Arbitrum │  │  ETH   │  │   Sui   │   │   │
│  │    │Verifier │  │Verifier │  │ Verifier │  │Verifier│  │Verifier │   │   │
│  │    └─────────┘  └─────────┘  └──────────┘  └────────┘  └─────────┘   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Infrastructure

### 1.1 Project Setup

**Directory Structure:**
```
spendingproofs/
├── Cargo.toml                    # Workspace root
├── crates/
│   ├── prover/                   # Core proving service
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── policy.rs         # Policy loading & management
│   │       ├── prover.rs         # Proof generation wrapper
│   │       └── types.rs          # Shared types
│   │
│   ├── api/                      # HTTP API service
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs           # Axum server entry
│   │       ├── routes/
│   │       │   ├── mod.rs
│   │       │   ├── prove.rs      # POST /v1/prove
│   │       │   ├── verify.rs     # POST /v1/verify
│   │       │   └── policies.rs   # GET /v1/policies
│   │       ├── middleware.rs     # Auth, rate limiting
│   │       └── ws.rs             # WebSocket for progress
│   │
│   └── cli/                      # CLI for local proving
│       ├── Cargo.toml
│       └── src/main.rs
│
├── policies/                     # Pre-built policy ONNX models
│   ├── spending-basic.onnx
│   ├── spending-advanced.onnx
│   ├── rate-limit.onnx
│   ├── access-control.onnx
│   └── manifests/                # Policy metadata
│       └── spending-basic.json
│
├── contracts/                    # Solidity verifier contracts
│   ├── src/
│   │   ├── PolicyVerifier.sol    # Core verifier
│   │   ├── ProofRegistry.sol     # Proof storage/lookup
│   │   └── interfaces/
│   ├── script/                   # Deployment scripts
│   ├── test/
│   └── foundry.toml
│
├── demo/                         # Interactive demo UI
│   ├── package.json
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   │   ├── PolicySelector.tsx
│   │   │   ├── InputForm.tsx
│   │   │   ├── ProofProgress.tsx
│   │   │   ├── ChainVerifier.tsx
│   │   │   └── ProofArtifact.tsx
│   │   └── lib/
│   │       └── api.ts
│   └── next.config.js
│
├── models/                       # Policy model training
│   ├── notebooks/
│   │   ├── spending_policy.ipynb
│   │   └── export_onnx.ipynb
│   └── requirements.txt
│
└── docker/
    ├── Dockerfile.prover
    ├── Dockerfile.api
    └── docker-compose.yml
```

### 1.2 Rust Workspace Configuration

**Root Cargo.toml:**
```toml
[workspace]
resolver = "2"
members = [
    "crates/prover",
    "crates/api",
    "crates/cli",
]

[workspace.dependencies]
# jolt-atlas dependencies
jolt-core = { git = "https://github.com/ICME-Lab/zkml-jolt" }
onnx-tracer = { git = "https://github.com/ICME-Lab/jolt-atlas" }
zkml-jolt-core = { git = "https://github.com/ICME-Lab/jolt-atlas" }

# Crypto
ark-bn254 = "0.5.0"
ark-ff = "0.5.0"
ark-serialize = "0.5.0"

# API
axum = { version = "0.7", features = ["ws", "json"] }
tokio = { version = "1", features = ["full"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Utilities
thiserror = "1.0"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"
```

### 1.3 Core Prover Crate

**crates/prover/src/lib.rs:**
```rust
//! Spending Proofs - Policy proof generation using jolt-atlas

mod policy;
mod prover;
mod types;

pub use policy::{Policy, PolicyRegistry};
pub use prover::{ProofGenerator, ProofResult};
pub use types::{PolicyInput, PolicyOutput, ProofArtifact};
```

**crates/prover/src/types.rs:**
```rust
use serde::{Deserialize, Serialize};

/// Input to a policy decision
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PolicyInput {
    Spending(SpendingInput),
    RateLimit(RateLimitInput),
    AccessControl(AccessControlInput),
    Custom(serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpendingInput {
    pub price_usdc: f64,
    pub budget_usdc: f64,
    pub daily_spent: f64,
    pub daily_limit: f64,
    pub service_reputation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitInput {
    pub calls_today: u32,
    pub tier: String,
    pub endpoint_cost: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlInput {
    pub role: String,
    pub resource: String,
    pub action: String,
    pub context: serde_json::Value,
}

/// Output from a policy decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyOutput {
    pub decision: bool,
    pub confidence: f64,
    pub reason_code: Option<String>,
}

/// Generated proof artifact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofArtifact {
    pub proof: Vec<u8>,
    pub proof_hex: String,
    pub proof_size_bytes: usize,
    pub generation_time_ms: u64,
    pub model_hash: String,
    pub input_commitment: String,
    pub output_commitment: String,
    pub program_io: ProgramIOJson,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramIOJson {
    pub input: Vec<i32>,
    pub output: Vec<i32>,
}
```

**crates/prover/src/policy.rs:**
```rust
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// Policy metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyManifest {
    pub id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub input_schema: serde_json::Value,
    pub output_schema: serde_json::Value,
    pub onnx_path: PathBuf,
    pub memory_size: usize,  // e.g., 1 << 14 for 2^14
    pub input_scale: i32,
    pub param_scale: i32,
}

/// A loaded policy ready for proving
pub struct Policy {
    pub manifest: PolicyManifest,
    pub model: onnx_tracer::Model,
}

/// Registry of available policies
pub struct PolicyRegistry {
    policies: HashMap<String, PolicyManifest>,
    policies_dir: PathBuf,
}

impl PolicyRegistry {
    pub fn new(policies_dir: impl AsRef<Path>) -> Result<Self> {
        let mut policies = HashMap::new();
        let policies_dir = policies_dir.as_ref().to_path_buf();

        // Load all manifest files
        let manifests_dir = policies_dir.join("manifests");
        for entry in std::fs::read_dir(&manifests_dir)? {
            let entry = entry?;
            if entry.path().extension().map_or(false, |e| e == "json") {
                let manifest: PolicyManifest =
                    serde_json::from_reader(std::fs::File::open(entry.path())?)?;
                policies.insert(manifest.id.clone(), manifest);
            }
        }

        Ok(Self { policies, policies_dir })
    }

    pub fn list(&self) -> Vec<&PolicyManifest> {
        self.policies.values().collect()
    }

    pub fn get(&self, id: &str) -> Option<&PolicyManifest> {
        self.policies.get(id)
    }

    pub fn load(&self, id: &str) -> Result<Policy> {
        let manifest = self.get(id)
            .ok_or_else(|| anyhow::anyhow!("Policy not found: {}", id))?
            .clone();

        let onnx_path = self.policies_dir.join(&manifest.onnx_path);
        let model = onnx_tracer::model(onnx_path.to_str().unwrap());

        Ok(Policy { manifest, model })
    }
}
```

**crates/prover/src/prover.rs:**
```rust
use crate::types::{PolicyInput, PolicyOutput, ProofArtifact, ProgramIOJson};
use crate::policy::Policy;
use anyhow::Result;
use std::time::Instant;

use zkml_jolt_core::jolt::{
    JoltSNARK,
    pcs::dory::DoryKeccak,
    field::JoltField,
};
use ark_bn254::Fr;
use ark_serialize::CanonicalSerialize;

/// Proof generator wrapping jolt-atlas
pub struct ProofGenerator {
    // Preprocessed parameters could be cached here
}

impl ProofGenerator {
    pub fn new() -> Self {
        Self {}
    }

    /// Generate a proof for a policy decision
    pub fn prove(
        &self,
        policy: &Policy,
        input: &PolicyInput,
        expected_output: &PolicyOutput,
    ) -> Result<ProofArtifact> {
        let start = Instant::now();

        // Convert input to fixed-point i32 vector
        let input_vec = self.encode_input(input, &policy.manifest)?;

        // Preprocess the model
        let (pp, vp) = JoltSNARK::<Fr, DoryKeccak, _>::prover_preprocess(
            &policy.model,
            1 << policy.manifest.memory_size,
        );

        // Generate the proof
        let (proof, program_io) = JoltSNARK::prove(
            &pp,
            &policy.model,
            input_vec.clone(),
        );

        let generation_time = start.elapsed().as_millis() as u64;

        // Serialize the proof
        let mut proof_bytes = Vec::new();
        proof.serialize_compressed(&mut proof_bytes)?;

        // Compute hashes for commitments
        let model_hash = self.compute_model_hash(&policy.model)?;
        let input_commitment = self.compute_commitment(&input_vec)?;
        let output_commitment = self.compute_commitment(&program_io.output)?;

        Ok(ProofArtifact {
            proof: proof_bytes.clone(),
            proof_hex: hex::encode(&proof_bytes),
            proof_size_bytes: proof_bytes.len(),
            generation_time_ms: generation_time,
            model_hash,
            input_commitment,
            output_commitment,
            program_io: ProgramIOJson {
                input: program_io.input,
                output: program_io.output,
            },
        })
    }

    /// Verify a proof locally
    pub fn verify(
        &self,
        policy: &Policy,
        artifact: &ProofArtifact,
    ) -> Result<bool> {
        // Deserialize and verify
        let (_, vp) = JoltSNARK::<Fr, DoryKeccak, _>::prover_preprocess(
            &policy.model,
            1 << policy.manifest.memory_size,
        );

        let proof = JoltSNARK::deserialize(&artifact.proof)?;
        let program_io = /* reconstruct from artifact */;

        Ok(JoltSNARK::verify(&vp, &proof, &program_io).is_ok())
    }

    fn encode_input(&self, input: &PolicyInput, manifest: &PolicyManifest) -> Result<Vec<i32>> {
        // Convert floating point inputs to fixed-point based on scale
        let scale = manifest.input_scale as f64;

        match input {
            PolicyInput::Spending(s) => {
                Ok(vec![
                    (s.price_usdc * scale) as i32,
                    (s.budget_usdc * scale) as i32,
                    (s.daily_spent * scale) as i32,
                    (s.daily_limit * scale) as i32,
                    (s.service_reputation * scale) as i32,
                ])
            }
            // ... other input types
            _ => anyhow::bail!("Unsupported input type"),
        }
    }

    fn compute_model_hash(&self, model: &onnx_tracer::Model) -> Result<String> {
        // Hash the model bytes
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        // hasher.update(&model.bytes);
        Ok(format!("0x{}", hex::encode(hasher.finalize())))
    }

    fn compute_commitment(&self, data: &[i32]) -> Result<String> {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        for &val in data {
            hasher.update(&val.to_le_bytes());
        }
        Ok(format!("0x{}", hex::encode(hasher.finalize())))
    }
}
```

---

## Phase 2: API Service

### 2.1 HTTP API Implementation

**crates/api/src/main.rs:**
```rust
use axum::{
    routing::{get, post},
    Router,
    Json,
    extract::{State, ws::WebSocket},
};
use tower_http::cors::CorsLayer;
use std::sync::Arc;
use prover::{PolicyRegistry, ProofGenerator};

mod routes;
mod ws;

#[derive(Clone)]
pub struct AppState {
    pub registry: Arc<PolicyRegistry>,
    pub generator: Arc<ProofGenerator>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::init();

    let registry = PolicyRegistry::new("./policies")?;
    let generator = ProofGenerator::new();

    let state = AppState {
        registry: Arc::new(registry),
        generator: Arc::new(generator),
    };

    let app = Router::new()
        .route("/v1/prove", post(routes::prove::generate_proof))
        .route("/v1/prove/stream", get(routes::prove::stream_proof))
        .route("/v1/verify", post(routes::verify::verify_proof))
        .route("/v1/policies", get(routes::policies::list_policies))
        .route("/v1/policies/:id", get(routes::policies::get_policy))
        .route("/v1/contracts", get(routes::contracts::list_contracts))
        .route("/health", get(|| async { "OK" }))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    tracing::info!("API server listening on :3000");
    axum::serve(listener, app).await?;

    Ok(())
}
```

**crates/api/src/routes/prove.rs:**
```rust
use axum::{
    extract::{State, ws::{WebSocket, WebSocketUpgrade}},
    Json,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use crate::AppState;
use prover::{PolicyInput, PolicyOutput, ProofArtifact};

#[derive(Debug, Deserialize)]
pub struct ProveRequest {
    pub policy_model: String,
    pub inputs: PolicyInput,
    pub outputs: PolicyOutput,
}

#[derive(Debug, Serialize)]
pub struct ProveResponse {
    pub proof: String,
    pub proof_size_bytes: usize,
    pub generation_time_ms: u64,
    pub model_hash: String,
    pub input_commitment: String,
    pub output_commitment: String,
    pub verification_contracts: VerificationContracts,
}

#[derive(Debug, Serialize)]
pub struct VerificationContracts {
    pub ethereum: Option<String>,
    pub base: Option<String>,
    pub arbitrum: Option<String>,
    pub avalanche: Option<String>,
}

pub async fn generate_proof(
    State(state): State<AppState>,
    Json(req): Json<ProveRequest>,
) -> Result<Json<ProveResponse>, axum::http::StatusCode> {
    // Load the policy
    let policy = state.registry.load(&req.policy_model)
        .map_err(|_| axum::http::StatusCode::NOT_FOUND)?;

    // Generate the proof (this blocks, consider spawning)
    let artifact = tokio::task::spawn_blocking(move || {
        state.generator.prove(&policy, &req.inputs, &req.outputs)
    })
    .await
    .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?
    .map_err(|_| axum::http::StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(Json(ProveResponse {
        proof: artifact.proof_hex,
        proof_size_bytes: artifact.proof_size_bytes,
        generation_time_ms: artifact.generation_time_ms,
        model_hash: artifact.model_hash,
        input_commitment: artifact.input_commitment,
        output_commitment: artifact.output_commitment,
        verification_contracts: VerificationContracts {
            ethereum: Some("0x...".to_string()),
            base: Some("0x...".to_string()),
            arbitrum: Some("0x...".to_string()),
            avalanche: Some("0x...".to_string()),
        },
    }))
}

/// WebSocket endpoint for streaming proof progress
pub async fn stream_proof(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_proof_stream(socket, state))
}

async fn handle_proof_stream(mut socket: WebSocket, state: AppState) {
    // Receive the request
    // Stream progress updates as JSON messages
    // Send final proof when complete
}
```

### 2.2 API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /v1/prove` | POST | Generate a proof for a policy decision |
| `WS /v1/prove/stream` | WS | Stream proof generation with progress updates |
| `POST /v1/verify` | POST | Verify a proof locally |
| `GET /v1/policies` | GET | List available policy templates |
| `GET /v1/policies/:id` | GET | Get details for a specific policy |
| `GET /v1/contracts` | GET | Get verification contract addresses per chain |
| `GET /health` | GET | Health check endpoint |

---

## Phase 3: Policy Templates

### 3.1 Policy Model Training

Each policy is a simple neural network trained to make binary decisions. The network architecture:

```
Input Layer (N features) → Dense(32) → ReLU → Dense(16) → ReLU → Dense(2) → Softmax
```

**models/notebooks/spending_policy.ipynb** will contain:

```python
import torch
import torch.nn as nn

class SpendingPolicy(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        logits = self.layers(x)
        return torch.softmax(logits, dim=-1)

# Training data encodes the policy rules:
# - price <= budget
# - daily_spent + price <= daily_limit
# - service_reputation >= 0.5 (or higher threshold for larger amounts)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "spending-basic.onnx",
    input_names=["features"],
    output_names=["decision"],
    opset_version=12,
)
```

### 3.2 Policy Manifest Format

**policies/manifests/spending-basic.json:**
```json
{
  "id": "spending-basic",
  "name": "Basic Spending Policy",
  "description": "Simple budget and daily limit checks for AI agent spending",
  "version": "1.0.0",
  "onnx_path": "spending-basic.onnx",
  "memory_size": 14,
  "input_scale": 128,
  "param_scale": 7,
  "input_schema": {
    "type": "object",
    "properties": {
      "price_usdc": { "type": "number", "description": "Transaction price in USDC" },
      "budget_usdc": { "type": "number", "description": "Total budget available" },
      "daily_spent": { "type": "number", "description": "Amount spent today" },
      "daily_limit": { "type": "number", "description": "Daily spending limit" },
      "service_reputation": { "type": "number", "minimum": 0, "maximum": 1 }
    },
    "required": ["price_usdc", "budget_usdc", "daily_spent", "daily_limit", "service_reputation"]
  },
  "output_schema": {
    "type": "object",
    "properties": {
      "decision": { "type": "boolean" },
      "confidence": { "type": "number" }
    }
  }
}
```

### 3.3 Policy Template Library

| Policy ID | Description | Inputs | Use Case |
|-----------|-------------|--------|----------|
| `spending-basic` | Simple budget limits | price, budget, daily_limit | Basic agent spending |
| `spending-advanced` | Reputation-aware spending | + service_score, category, time | x402 payments |
| `rate-limit` | API call gating | calls_today, tier, endpoint_cost | API usage control |
| `access-control` | Permission decisions | role, resource, action, context | Authorization |
| `compliance-check` | Regulatory gates | jurisdiction, amount, risk_score | Financial compliance |

---

## Phase 4: Solidity Verification Contracts

### 4.1 Verifier Contract

**contracts/src/PolicyVerifier.sol:**
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {Groth16Verifier} from "./Groth16Verifier.sol";

/// @title PolicyVerifier
/// @notice Verifies zkML policy proofs on-chain
contract PolicyVerifier {
    /// @notice Emitted when a proof is successfully verified
    event ProofVerified(
        bytes32 indexed policyHash,
        bytes32 inputCommitment,
        bytes32 outputCommitment,
        bool decision,
        uint256 timestamp
    );

    /// @notice Mapping of policy hashes to their verification keys
    mapping(bytes32 => bytes) public verificationKeys;

    /// @notice Registry of verified proofs
    mapping(bytes32 => bool) public verifiedProofs;

    /// @notice Owner for key registration
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    /// @notice Register a verification key for a policy
    function registerPolicy(
        bytes32 policyHash,
        bytes calldata verificationKey
    ) external {
        require(msg.sender == owner, "Unauthorized");
        verificationKeys[policyHash] = verificationKey;
    }

    /// @notice Verify a policy proof
    /// @param policyHash Hash of the policy model
    /// @param proof The zkSNARK proof bytes
    /// @param inputCommitment Commitment to the inputs
    /// @param outputCommitment Commitment to the outputs
    /// @param decision The policy decision (approve/deny)
    function verifyProof(
        bytes32 policyHash,
        bytes calldata proof,
        bytes32 inputCommitment,
        bytes32 outputCommitment,
        bool decision
    ) external returns (bool) {
        // Verify the proof using the registered verification key
        bytes memory vk = verificationKeys[policyHash];
        require(vk.length > 0, "Policy not registered");

        // Construct public inputs
        uint256[4] memory publicInputs = [
            uint256(policyHash),
            uint256(inputCommitment),
            uint256(outputCommitment),
            decision ? 1 : 0
        ];

        // Verify (implementation depends on jolt-atlas verifier structure)
        bool valid = _verifyGroth16(proof, publicInputs, vk);
        require(valid, "Invalid proof");

        // Record the verification
        bytes32 proofId = keccak256(abi.encodePacked(
            policyHash,
            inputCommitment,
            outputCommitment,
            decision
        ));
        verifiedProofs[proofId] = true;

        emit ProofVerified(
            policyHash,
            inputCommitment,
            outputCommitment,
            decision,
            block.timestamp
        );

        return true;
    }

    /// @notice Check if a proof has been verified
    function isProofVerified(bytes32 proofId) external view returns (bool) {
        return verifiedProofs[proofId];
    }

    function _verifyGroth16(
        bytes calldata proof,
        uint256[4] memory publicInputs,
        bytes memory vk
    ) internal pure returns (bool) {
        // Actual verification logic using pairing checks
        // This would use precompiled contracts for BN254 pairing
        return true; // Placeholder
    }
}
```

### 4.2 Proof Registry Contract

**contracts/src/ProofRegistry.sol:**
```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import {PolicyVerifier} from "./PolicyVerifier.sol";

/// @title ProofRegistry
/// @notice Stores and indexes verified policy proofs
contract ProofRegistry {
    struct VerifiedProof {
        bytes32 policyHash;
        bytes32 inputCommitment;
        bytes32 outputCommitment;
        bool decision;
        uint256 timestamp;
        address submitter;
    }

    /// @notice All verified proofs
    mapping(bytes32 => VerifiedProof) public proofs;

    /// @notice Proofs by policy
    mapping(bytes32 => bytes32[]) public proofsByPolicy;

    /// @notice Proofs by submitter
    mapping(address => bytes32[]) public proofsBySubmitter;

    /// @notice The verifier contract
    PolicyVerifier public verifier;

    constructor(address _verifier) {
        verifier = PolicyVerifier(_verifier);
    }

    /// @notice Submit and verify a proof
    function submitProof(
        bytes32 policyHash,
        bytes calldata proof,
        bytes32 inputCommitment,
        bytes32 outputCommitment,
        bool decision
    ) external returns (bytes32 proofId) {
        // Verify the proof
        require(
            verifier.verifyProof(
                policyHash,
                proof,
                inputCommitment,
                outputCommitment,
                decision
            ),
            "Proof verification failed"
        );

        // Store the proof
        proofId = keccak256(abi.encodePacked(
            policyHash,
            inputCommitment,
            outputCommitment,
            decision,
            block.timestamp,
            msg.sender
        ));

        proofs[proofId] = VerifiedProof({
            policyHash: policyHash,
            inputCommitment: inputCommitment,
            outputCommitment: outputCommitment,
            decision: decision,
            timestamp: block.timestamp,
            submitter: msg.sender
        });

        proofsByPolicy[policyHash].push(proofId);
        proofsBySubmitter[msg.sender].push(proofId);

        return proofId;
    }

    /// @notice Get proof count for a policy
    function getProofCount(bytes32 policyHash) external view returns (uint256) {
        return proofsByPolicy[policyHash].length;
    }
}
```

### 4.3 Deployment Strategy

Deploy to all target chains with same addresses (using CREATE2):

| Chain | Verifier Address | Registry Address |
|-------|------------------|------------------|
| Ethereum Mainnet | 0x... | 0x... |
| Base | 0x... | 0x... |
| Arbitrum One | 0x... | 0x... |
| Avalanche C-Chain | 0x... | 0x... |
| Optimism | 0x... | 0x... |

---

## Phase 5: Demo UI

### 5.1 Next.js App Structure

**demo/src/app/page.tsx:**
```tsx
import { PolicySelector } from '@/components/PolicySelector';
import { InputForm } from '@/components/InputForm';
import { ProofProgress } from '@/components/ProofProgress';
import { ChainVerifier } from '@/components/ChainVerifier';
import { ProofArtifact } from '@/components/ProofArtifact';

export default function Home() {
  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-8">
        Policy Proofs Demo
      </h1>

      <div className="grid gap-6">
        <PolicySelector />
        <InputForm />
        <ProofProgress />
        <ChainVerifier />
        <ProofArtifact />
      </div>
    </main>
  );
}
```

### 5.2 Key Components

**PolicySelector.tsx:**
- Radio buttons for preset policies
- Option to upload custom ONNX model
- Shows policy description and input schema

**InputForm.tsx:**
- Dynamic form based on selected policy's input schema
- Sliders for numeric values with min/max from schema
- Real-time validation

**ProofProgress.tsx:**
- Progress bar showing proof generation stages
- Elapsed time counter
- Estimated time remaining
- Shows intermediate values (model hash, commitments)

**ChainVerifier.tsx:**
- Chain selector dropdown
- "Verify On-Chain" button
- Transaction status and explorer link
- Verification time display

**ProofArtifact.tsx:**
- Displays proof hex (truncated)
- Download proof as file
- Copy to clipboard
- Proof size in bytes

### 5.3 WebSocket Integration

```typescript
// demo/src/lib/api.ts

export interface ProofProgress {
  stage: 'preprocessing' | 'tracing' | 'proving' | 'finalizing';
  progress: number; // 0-100
  elapsed_ms: number;
  estimated_total_ms: number;
}

export async function streamProof(
  request: ProveRequest,
  onProgress: (progress: ProofProgress) => void
): Promise<ProofResponse> {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket('ws://localhost:3000/v1/prove/stream');

    ws.onopen = () => {
      ws.send(JSON.stringify(request));
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'progress') {
        onProgress(data.payload);
      } else if (data.type === 'complete') {
        resolve(data.payload);
        ws.close();
      } else if (data.type === 'error') {
        reject(new Error(data.message));
        ws.close();
      }
    };

    ws.onerror = (error) => {
      reject(error);
    };
  });
}
```

---

## Phase 6: Docker & Deployment

### 6.1 Dockerfiles

**docker/Dockerfile.prover:**
```dockerfile
FROM rust:1.75-bookworm as builder

WORKDIR /app
COPY . .

RUN cargo build --release --package api

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/api /usr/local/bin/
COPY --from=builder /app/policies /policies

ENV POLICIES_DIR=/policies
EXPOSE 3000

CMD ["api"]
```

**docker/docker-compose.yml:**
```yaml
version: '3.8'

services:
  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.prover
    ports:
      - "3000:3000"
    environment:
      - RUST_LOG=info
      - POLICIES_DIR=/policies
    volumes:
      - ../policies:/policies:ro

  demo:
    build:
      context: ../demo
    ports:
      - "3001:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://api:3000
    depends_on:
      - api
```

---

## Implementation Timeline

### Week 1: Foundation
- [ ] Set up Rust workspace structure
- [ ] Integrate jolt-atlas as dependency
- [ ] Implement basic prover crate with one policy
- [ ] Create first policy ONNX model (spending-basic)
- [ ] Basic CLI for local proof generation

### Week 2: API Layer
- [ ] Implement Axum API server
- [ ] Add all REST endpoints
- [ ] Implement WebSocket streaming
- [ ] Add request validation and error handling
- [ ] Docker containerization

### Week 3: Policies & Contracts
- [ ] Train and export remaining policy models
- [ ] Create policy manifest format and loader
- [ ] Implement Solidity verifier contract
- [ ] Deploy contracts to testnets
- [ ] Integration tests

### Week 4: Demo UI
- [ ] Set up Next.js project
- [ ] Implement all UI components
- [ ] WebSocket integration
- [ ] Chain verification integration
- [ ] Polish and documentation

---

## Success Metrics

1. **Proof Generation**: < 10 seconds for basic policies
2. **Proof Size**: < 100KB per proof
3. **Verification Time**: < 200ms on-chain
4. **API Latency**: < 50ms excluding proof generation
5. **Demo UX**: Proof generation visible with progress updates

---

## Open Questions

1. **Verification Key Generation**: How does jolt-atlas export verification keys for Solidity contracts?
2. **Proof Format**: What's the exact proof structure for on-chain verification?
3. **Memory Requirements**: What's the minimum memory for the prover service?
4. **Custom ONNX Support**: What ONNX operations are supported/unsupported?

---

## Next Steps

1. Clone jolt-atlas and run the authorization example locally
2. Understand the proof serialization format
3. Design the verification key export mechanism
4. Begin with Phase 1.1 (Project Setup)
