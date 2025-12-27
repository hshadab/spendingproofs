# Arc Spending Policy Proofs

**zkML Spending Proofs for USDC Agents on Arc**

Cryptographic proof of policy compliance for autonomous USDC agents. Prove your agent followed its spending rules—without revealing them. Built on Jolt-Atlas. Designed for Arc.

## Live Demo

**https://hshadab.github.io/spendingproofs/**

## What is this?

Spending Proofs is a trust primitive for agentic commerce on Arc. It enables autonomous agents to generate SNARK proofs that cryptographically attest they followed their spending policy before making purchases.

### Key Features

- **zkML Policy Proofs**: Every spending decision generates a zero-knowledge proof
- **8-Input Spending Model**: Price, budget, daily limits, service reputation, behavioral patterns
- **3-Output Decision**: shouldBuy (boolean), confidence (0-100%), riskScore (0-100)
- **Tamper Detection**: Cryptographic input hashing prevents post-proof modification
- **Arc-Native**: Designed for Arc's sub-second finality and USDC gas

## Why Arc?

Autonomous agents require Arc's unique attributes:

| Requirement | Why It's Necessary |
|-------------|-------------------|
| **USDC Gas** | Agents budget in stablecoins. Volatile gas breaks autonomous operation. |
| **Deterministic Finality** | Sub-second finality enables real-time agent commerce. No reorgs. |
| **Opt-in Privacy** | Protect agent strategies from front-running and MEV. |
| **Enterprise Rails** | Circle-backed infrastructure for production deployment. |

## Tech Stack

- **Frontend**: Next.js 15, React 18, Tailwind CSS
- **zkML Prover**: Jolt-Atlas (HyperKZG + BN254)
- **Blockchain**: Arc Testnet (Chain ID: 5042002)
- **Deployment**: GitHub Pages (static export)

## Local Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## Project Structure

```
src/
├── app/
│   ├── page.tsx              # Landing page
│   ├── demo/
│   │   ├── playground/       # Interactive proof generation
│   │   ├── payment/          # End-to-end payment flow
│   │   └── tamper/           # Tamper detection demo
├── components/
│   ├── PolicySliders.tsx     # Policy configuration UI
│   ├── PurchaseSimulator.tsx # Input simulation
│   ├── ProofProgress.tsx     # Proof generation progress
│   ├── ProofViewer.tsx       # Proof metadata display
│   └── TamperPanel.tsx       # Tamper detection demo
├── hooks/
│   └── useProofGeneration.ts # Proof generation hook
└── lib/
    ├── spendingModel.ts      # Spending policy model
    ├── contracts.ts          # Arc contract addresses
    └── types.ts              # TypeScript types
```

## Spending Model

The spending policy model evaluates 8 inputs:

| Input | Description |
|-------|-------------|
| `priceUsdc` | Purchase amount |
| `budgetUsdc` | Remaining budget |
| `spentTodayUsdc` | Today's spending |
| `dailyLimitUsdc` | Policy daily limit |
| `serviceSuccessRate` | Service reputation (0-1) |
| `serviceTotalCalls` | Service call history |
| `purchasesInCategory` | Category purchase count |
| `timeSinceLastPurchase` | Hours since last purchase |

And produces 3 outputs:

| Output | Description |
|--------|-------------|
| `shouldBuy` | Binary decision |
| `confidence` | Decision confidence (0-100%) |
| `riskScore` | Risk assessment (0-100) |

## Disclaimer

This project demonstrates an Arc-native design pattern for zkML spending proofs. It is not affiliated with Circle Internet Financial or the official Arc blockchain team. The package namespace `@icme-labs/spending-proofs` is used for this demonstration.

## License

MIT
