#!/usr/bin/env node

/**
 * Arc Policy Proofs CLI
 *
 * Generate and verify zkML spending proofs from the command line.
 *
 * Usage:
 *   arc-prove prove --price 0.05 --budget 1.0 --spent 0.2 --limit 0.5
 *   arc-prove verify <proof-hash> --inputs '{"priceUsdc": 0.05, ...}'
 *   arc-prove health
 */

import { Command } from 'commander';

const DEFAULT_PROVER_URL = 'http://localhost:3001';

// API response types
interface ProveApiResponse {
  success: boolean;
  error?: string;
  proof: {
    proof: string;
    proof_hash: string;
  };
  inference: {
    raw_output: number[];
  };
}

interface HealthApiResponse {
  models?: string[];
}

interface RpcResponse {
  result?: string;
  error?: { message: string };
}

interface SpendingInput {
  priceUsdc: number;
  budgetUsdc: number;
  spentTodayUsdc: number;
  dailyLimitUsdc: number;
  serviceSuccessRate: number;
  serviceTotalCalls: number;
  purchasesInCategory: number;
  timeSinceLastPurchase: number;
}

function spendingInputToArray(input: SpendingInput): number[] {
  return [
    input.priceUsdc,
    input.budgetUsdc,
    input.spentTodayUsdc,
    input.dailyLimitUsdc,
    input.serviceSuccessRate,
    input.serviceTotalCalls,
    input.purchasesInCategory,
    input.timeSinceLastPurchase,
  ];
}

const program = new Command();

program
  .name('arc-prove')
  .description('Generate and verify zkML spending proofs for Arc chain')
  .version('0.1.0');

program
  .command('prove')
  .description('Generate a SNARK proof for a spending decision')
  .option('--price <usdc>', 'Purchase price in USDC', parseFloat)
  .option('--budget <usdc>', 'Available budget in USDC', parseFloat)
  .option('--spent <usdc>', 'Amount spent today in USDC', parseFloat)
  .option('--limit <usdc>', 'Daily spending limit in USDC', parseFloat)
  .option('--success-rate <rate>', 'Service success rate (0-1)', parseFloat, 0.95)
  .option('--total-calls <n>', 'Total service calls', parseInt, 100)
  .option('--category-purchases <n>', 'Purchases in category', parseInt, 5)
  .option('--hours-since-last <h>', 'Hours since last purchase', parseFloat, 2)
  .option('--prover <url>', 'Prover service URL', DEFAULT_PROVER_URL)
  .option('--json', 'Output as JSON')
  .action(async (options) => {
    if (!options.price || !options.budget || options.spent === undefined || !options.limit) {
      console.error('Error: --price, --budget, --spent, and --limit are required');
      process.exit(1);
    }

    const input: SpendingInput = {
      priceUsdc: options.price,
      budgetUsdc: options.budget,
      spentTodayUsdc: options.spent,
      dailyLimitUsdc: options.limit,
      serviceSuccessRate: options.successRate,
      serviceTotalCalls: options.totalCalls,
      purchasesInCategory: options.categoryPurchases,
      timeSinceLastPurchase: options.hoursSinceLast,
    };

    if (!options.json) {
      console.log('\n🔐 Generating zkML proof...\n');
      console.log('Inputs:');
      console.log(`  Price: $${input.priceUsdc.toFixed(2)}`);
      console.log(`  Budget: $${input.budgetUsdc.toFixed(2)}`);
      console.log(`  Spent today: $${input.spentTodayUsdc.toFixed(2)}`);
      console.log(`  Daily limit: $${input.dailyLimitUsdc.toFixed(2)}`);
      console.log(`  Service success rate: ${(input.serviceSuccessRate * 100).toFixed(0)}%`);
      console.log(`  Total calls: ${input.serviceTotalCalls}`);
      console.log(`  Category purchases: ${input.purchasesInCategory}`);
      console.log(`  Hours since last: ${input.timeSinceLastPurchase}`);
      console.log('\nGenerating SNARK proof (this may take 4-12 seconds)...\n');
    }

    try {
      const startTime = Date.now();
      const response = await fetch(`${options.prover}/prove`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_id: 'spending-model',
          inputs: spendingInputToArray(input),
          tag: 'spending',
        }),
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(`Prover error: ${error}`);
      }

      const data = (await response.json()) as ProveApiResponse;
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

      if (!data.success) {
        throw new Error(data.error || 'Proof generation failed');
      }

      if (options.json) {
        console.log(JSON.stringify({
          success: true,
          decision: {
            shouldBuy: data.inference.raw_output[0] > 0.5,
            confidence: data.inference.raw_output[1],
            riskScore: data.inference.raw_output[2],
          },
          proof: {
            hash: data.proof.proof_hash,
            size: data.proof.proof?.length || 0,
            generationTime: parseFloat(elapsed),
          },
          raw: data,
        }, null, 2));
      } else {
        const decision = data.inference.raw_output[0] > 0.5 ? '✅ APPROVE' : '❌ REJECT';
        const confidence = (data.inference.raw_output[1] * 100).toFixed(0);
        const risk = (data.inference.raw_output[2] * 100).toFixed(0);

        console.log(`Decision: ${decision}`);
        console.log(`Confidence: ${confidence}%`);
        console.log(`Risk Score: ${risk}%`);
        console.log(`\nProof generated in ${elapsed}s`);
        console.log(`Proof hash: ${data.proof.proof_hash}`);
        console.log(`Proof size: ${((data.proof.proof?.length || 0) / 1024).toFixed(1)} KB`);
      }
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      if (options.json) {
        console.log(JSON.stringify({ success: false, error: message }));
      } else {
        console.error(`\n❌ Error: ${message}`);
      }
      process.exit(1);
    }
  });

program
  .command('decide')
  .description('Run spending decision locally (no proof)')
  .option('--price <usdc>', 'Purchase price in USDC', parseFloat)
  .option('--budget <usdc>', 'Available budget in USDC', parseFloat)
  .option('--spent <usdc>', 'Amount spent today in USDC', parseFloat)
  .option('--limit <usdc>', 'Daily spending limit in USDC', parseFloat)
  .option('--success-rate <rate>', 'Service success rate (0-1)', parseFloat, 0.95)
  .option('--total-calls <n>', 'Total service calls', parseInt, 100)
  .option('--json', 'Output as JSON')
  .action((options) => {
    if (!options.price || !options.budget || options.spent === undefined || !options.limit) {
      console.error('Error: --price, --budget, --spent, and --limit are required');
      process.exit(1);
    }

    const withinBudget = options.price <= options.budget;
    const withinDailyLimit = options.spent + options.price <= options.limit;
    const serviceReliable = options.successRate >= 0.8 && options.totalCalls >= 10;

    const budgetScore = withinBudget ? 1.0 : 0.0;
    const limitScore = withinDailyLimit ? 1.0 : 0.0;
    const serviceScore = serviceReliable ? 1.0 : 0.5;

    const confidence = (budgetScore + limitScore + serviceScore) / 3;
    const shouldBuy = withinBudget && withinDailyLimit && confidence >= 0.7;
    const riskScore = 1 - confidence;

    if (options.json) {
      console.log(JSON.stringify({ shouldBuy, confidence, riskScore }));
    } else {
      console.log(`\nDecision: ${shouldBuy ? '✅ APPROVE' : '❌ REJECT'}`);
      console.log(`Confidence: ${(confidence * 100).toFixed(0)}%`);
      console.log(`Risk Score: ${(riskScore * 100).toFixed(0)}%`);
    }
  });

program
  .command('health')
  .description('Check prover service health')
  .option('--prover <url>', 'Prover service URL', DEFAULT_PROVER_URL)
  .option('--json', 'Output as JSON')
  .action(async (options) => {
    try {
      const response = await fetch(`${options.prover}/health`);
      const data = (await response.json()) as HealthApiResponse;

      if (options.json) {
        console.log(JSON.stringify({ healthy: response.ok, models: data.models }));
      } else {
        if (response.ok) {
          console.log('✅ Prover is healthy');
          if (data.models) {
            console.log(`   Models: ${data.models.join(', ')}`);
          }
        } else {
          console.log('❌ Prover is not responding');
        }
      }
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      if (options.json) {
        console.log(JSON.stringify({ healthy: false, error: message }));
      } else {
        console.error(`❌ Cannot connect to prover at ${options.prover}`);
      }
      process.exit(1);
    }
  });

program
  .command('check-attestation')
  .description('Check if a proof hash is attested on Arc chain (not cryptographic verification)')
  .argument('<proof-hash>', 'The proof hash to check')
  .option('--rpc <url>', 'RPC URL', 'https://rpc.testnet.arc.network')
  .option('--contract <address>', 'Proof attestation contract', '0xBE9a5DF7C551324CB872584C6E5bF56799787952')
  .option('--json', 'Output as JSON')
  .action(async (proofHash: string, options: { rpc: string; contract: string; json?: boolean }) => {
    try {
      const functionSelector = '0x8f742d16';
      const paddedHash = proofHash.replace('0x', '').padStart(64, '0');
      const callData = functionSelector + paddedHash;

      const response = await fetch(options.rpc, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          method: 'eth_call',
          params: [{ to: options.contract, data: callData }, 'latest'],
          id: 1,
        }),
      });

      const result = (await response.json()) as RpcResponse;
      const isAttested = result.result && result.result !== '0x' + '0'.repeat(64);

      if (options.json) {
        console.log(JSON.stringify({ attested: isAttested, proofHash }));
      } else {
        console.log(isAttested ? '✅ Proof is attested on-chain' : '❌ Proof not attested on-chain');
      }
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : 'Unknown error';
      if (options.json) {
        console.log(JSON.stringify({ valid: false, error: message }));
      } else {
        console.error(`❌ Error: ${message}`);
      }
      process.exit(1);
    }
  });

program.parse();
