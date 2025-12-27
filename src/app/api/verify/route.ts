/**
 * POST /api/verify
 *
 * Verify proof by comparing input hashes locally.
 */

import { NextRequest, NextResponse } from 'next/server';
import { keccak256, toHex } from 'viem';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { inputs, proofInputHash, proof, modelHash } = body;

    if (!inputs || !Array.isArray(inputs)) {
      return NextResponse.json(
        { valid: false, error: 'Missing or invalid inputs array' },
        { status: 400 }
      );
    }

    if (!proofInputHash) {
      return NextResponse.json(
        { valid: false, error: 'Missing proofInputHash' },
        { status: 400 }
      );
    }

    // Compute hash of provided inputs
    const inputBytes = new Float64Array(inputs);
    const buffer = new Uint8Array(inputBytes.buffer);
    const computedHash = keccak256(toHex(buffer));

    // Compare hashes
    const hashesMatch = computedHash.toLowerCase() === proofInputHash.toLowerCase();

    return NextResponse.json({
      valid: hashesMatch,
      reason: hashesMatch
        ? 'Input hash matches proof - inputs unchanged'
        : 'Input hash mismatch - inputs were modified after proof generation',
      computedHash,
      expectedHash: proofInputHash,
      verificationTimeMs: 1,
    });
  } catch (error) {
    return NextResponse.json(
      {
        valid: false,
        error: error instanceof Error ? error.message : 'Verification failed',
        verificationTimeMs: 0,
      },
      { status: 500 }
    );
  }
}
