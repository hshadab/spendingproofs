/**
 * Skyfire Agent API
 *
 * Create or retrieve a Skyfire agent identity with KYA credentials.
 * Skyfire provides agent identity verification (WHO is the agent).
 */

import { NextRequest, NextResponse } from 'next/server';
import { createAgent, getAgent, generateKYAToken } from '@/lib/skyfire';
import { createLogger } from '@/lib/metrics';

const logger = createLogger('api:skyfire:agent');

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { agentName, agentId, generateToken = false } = body;

    // If agentId provided, retrieve existing agent
    if (agentId) {
      logger.info('Retrieving Skyfire agent', { agentId });
      const agent = await getAgent(agentId);

      if (!agent) {
        return NextResponse.json(
          { success: false, error: 'Agent not found' },
          { status: 404 }
        );
      }

      let kyaToken = null;
      if (generateToken) {
        kyaToken = await generateKYAToken(agentId);
      }

      return NextResponse.json({
        success: true,
        agent,
        kyaToken,
      });
    }

    // Create new agent
    const name = agentName || 'zkML Demo Agent';
    logger.info('Creating Skyfire agent', { name });

    const agent = await createAgent(name);

    let kyaToken = null;
    if (generateToken) {
      kyaToken = await generateKYAToken(agent.id);
    }

    logger.info('Skyfire agent created', {
      agentId: agent.id,
      hasKyaToken: !!kyaToken,
    });

    return NextResponse.json({
      success: true,
      agent,
      kyaToken,
      message: 'Agent created with KYA (Know Your Agent) credentials',
    });
  } catch (error) {
    logger.error('Failed to create/retrieve agent', { error });
    return NextResponse.json(
      { success: false, error: 'Failed to manage agent identity' },
      { status: 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const agentId = searchParams.get('agentId');

  if (!agentId) {
    return NextResponse.json(
      { success: false, error: 'agentId required' },
      { status: 400 }
    );
  }

  try {
    const agent = await getAgent(agentId);

    if (!agent) {
      return NextResponse.json(
        { success: false, error: 'Agent not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({
      success: true,
      agent,
    });
  } catch (error) {
    logger.error('Failed to get agent', { error, agentId });
    return NextResponse.json(
      { success: false, error: 'Failed to retrieve agent' },
      { status: 500 }
    );
  }
}
