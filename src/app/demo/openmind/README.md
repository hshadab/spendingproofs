# OpenMind Robot Payments Demo

**Autonomous robots making USDC payments via x402 with zkML spending proofs**

> This demo showcases how OpenMind's OM1 robot operating system can integrate with
> zkML spending proofs to enable trustless autonomous commerce for embodied AI.

## Overview

OpenMind partnered with Circle in December 2025 to launch the first payment infrastructure
for autonomous agents and robots. This demo shows how zkML proofs can add a critical
trust layer to robot payments.

### The Problem

From OpenMind's own documentation:
> "Suitably configured system prompts will result in autonomous (and inherently unpredictable) real world actions by your robot"

When robots can autonomously spend money, owners need guarantees that spending policies
are actually being followed.

### The Solution

Before every x402 USDC payment, the robot generates a zkML proof that:
- The spending decision model was executed correctly
- All policy constraints were satisfied (limits, categories, reliability)
- The inputs weren't tampered with

**No proof, no payment.**

## Demo Flow

| Step | Description |
|------|-------------|
| 1. Robot Agent | Delivery robot running OM1 with USDC wallet |
| 2. Service Request | Robot needs to pay $0.50 for charging |
| 3. Policy Check | Evaluate against owner's spending limits |
| 4. zkML Proof | Generate 48KB SNARK proof of compliance |
| 5. x402 Payment | Execute USDC transfer with proof attached |
| 6. Audit Trail | Proof hash stored for compliance |

## Technical Details

### x402 Protocol

The x402 protocol (developed by Coinbase) enables internet-native payments using
the HTTP 402 "Payment Required" status code. OpenMind integrated x402 into OM1
to allow robots to pay for:

- Charging stations
- Navigation APIs (Mapbox, Google Maps)
- Cloud compute (AWS, GCP)
- Data services (weather, traffic)
- Transportation (Waymo)

### Robot Spending Policy

```typescript
interface RobotSpendingPolicy {
  dailyLimitUsdc: number;        // e.g., $10/day
  maxSingleTxUsdc: number;       // e.g., $2 per tx
  allowedCategories: string[];   // e.g., ['charging', 'navigation', 'compute']
  minServiceReliability: number; // e.g., 85%
  requireProofForAll: boolean;   // Always require zkML proof
}
```

### Proof Generation

| Metric | Value |
|--------|-------|
| Proof Size | ~48KB |
| Generation Time | 4-12s (real), <2s (simulated) |
| Prover | Jolt-Atlas SNARK |
| Verification | <150ms |

## Real vs Simulated Mode

The demo includes a toggle:
- **Simulated**: Fast mock proofs for quick demos
- **Real**: Calls the live Jolt-Atlas prover (4-12s)

## Use Cases

### 1. Delivery Robots
Autonomous delivery bots paying for charging, navigation APIs, and route optimization.

### 2. Inspection Drones
Industrial inspection drones paying for cloud compute for vision processing.

### 3. Home Assistants
Home robots paying for data services, smart home APIs, and maintenance.

### 4. Digital Agents
Pure software agents paying for API access, compute, and data.

## OpenMind + Circle Partnership

In December 2025, OpenMind and Circle announced:
- First USDC payment infrastructure for autonomous robots
- Micropayments as small as $0.004 USDC
- High-throughput batching for thousands of tx/second
- Pilot program launching in Silicon Valley

## Related Resources

- [OpenMind](https://openmind.org) - Open Source AI Robot Operating System
- [x402 Integration Docs](https://docs.openmind.org/robotics/coinbase-x402)
- [Circle](https://www.circle.com) - USDC Stablecoin
- [Main Spending Proofs Documentation](/docs/ARCHITECTURE.md)

---

**Part of the [NovaNet Jolt-Atlas Spending Proofs](/) project.**
