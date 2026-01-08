# Development Guide

This guide covers local development setup, testing, and contribution guidelines.

## Prerequisites

- **Node.js** 18+ (LTS recommended)
- **pnpm** or **npm** (pnpm preferred for workspace support)
- **Foundry** (for smart contract development)
- **Rust** (for prover development, optional)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/spendingproofs.git
cd spendingproofs

# Install dependencies
npm install

# Copy environment file
cp .env.example .env.local

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

## Environment Setup

### Required Environment Variables

Create `.env.local` with:

```bash
# Arc Testnet (defaults work for testnet)
NEXT_PUBLIC_ARC_RPC=https://rpc.testnet.arc.network

# Prover (optional - falls back to mock proofs)
PROVER_BACKEND_URL=http://localhost:3001

# For Crossmint integration (optional)
CROSSMINT_SERVER_KEY=your_staging_api_key

# For demo wallet transfers (testnet only!)
DEMO_WALLET_PRIVATE_KEY=0x...
```

### Optional Variables

```bash
# Signature authentication
REQUIRE_SIGNATURE_AUTH=false
ALLOWED_PROVER_ADDRESSES=0x123...,0x456...

# Cache settings
PROOF_CACHE_ENABLED=true
```

## Project Structure

```
spendingproofs/
├── src/                    # Next.js application source
│   ├── app/               # App Router pages and API routes
│   ├── components/        # React components
│   ├── hooks/             # Custom React hooks
│   ├── lib/               # Core libraries
│   └── __tests__/         # Unit tests
├── sdk/                    # NPM package source
│   └── src/               # SDK TypeScript source
├── contracts/              # Solidity smart contracts
│   ├── src/               # Contract source
│   └── test/              # Foundry tests
├── prover/                 # Rust prover binary
├── docs/                   # Documentation
└── scripts/               # Build and deployment scripts
```

## Development Commands

### Next.js App

```bash
# Development server with hot reload
npm run dev

# Type checking
npm run type-check

# Linting
npm run lint

# Format code
npm run format

# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Production build
npm run build
```

### SDK Development

```bash
# Build SDK
cd sdk && npm run build

# Link for local testing
cd sdk && npm link
cd .. && npm link @hshadab/spending-proofs

# Publish (requires npm access)
cd sdk && npm publish
```

### Smart Contracts

```bash
cd contracts

# Install dependencies
forge install

# Run tests
forge test

# Run tests with verbosity
forge test -vvv

# Deploy to testnet
forge script script/Deploy.s.sol --rpc-url arc_testnet --broadcast
```

## Testing

### Unit Tests

Tests are written with Vitest and located in `src/__tests__/`.

```bash
# Run all tests
npm test

# Run specific test file
npm test -- src/__tests__/spendingModel.test.ts

# Watch mode
npm test -- --watch

# Coverage report
npm run test:coverage
```

### Test Files

| File | Coverage |
|------|----------|
| `metrics.test.ts` | Metrics collection |
| `proofCache.test.ts` | Cache logic, invalidation |
| `retry.test.ts` | Retry and backoff |
| `signatureAuth.test.ts` | Signature verification |
| `spendingGate.test.ts` | Spending gate simulation |
| `spendingModel.test.ts` | Decision model |

### Contract Tests

```bash
cd contracts
forge test --match-test test_
```

## Code Style

### TypeScript

- Strict mode enabled
- ESLint with recommended rules
- Prettier for formatting

```bash
# Check linting
npm run lint

# Auto-fix lint issues
npm run lint -- --fix
```

### Naming Conventions

- **Files:** `camelCase.ts` for utilities, `PascalCase.tsx` for components
- **Functions:** `camelCase`
- **Types/Interfaces:** `PascalCase`
- **Constants:** `SCREAMING_SNAKE_CASE`

### Imports

Use path aliases:

```typescript
// Good
import { createLogger } from '@/lib/metrics';

// Avoid
import { createLogger } from '../../../lib/metrics';
```

## Debugging

### Structured Logging

All modules use `StructuredLogger`:

```typescript
import { createLogger } from '@/lib/metrics';

const logger = createLogger('myModule');

logger.info('Action completed', { action: 'doThing', duration: 100 });
logger.error('Failed', { error, action: 'doThing' });
```

Logs are JSON-formatted for easy parsing:

```json
{
  "timestamp": "2024-01-01T00:00:00.000Z",
  "level": "info",
  "component": "myModule",
  "message": "Action completed",
  "action": "doThing",
  "duration": 100
}
```

### React DevTools

Install React DevTools browser extension for component debugging.

### Network Debugging

Use browser DevTools Network tab to inspect API calls.

## Common Tasks

### Adding a New API Endpoint

1. Create route file in `src/app/api/[endpoint]/route.ts`
2. Import validation utilities from `@/lib/validation`
3. Import logger from `@/lib/metrics`
4. Follow standardized error response format

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { createLogger } from '@/lib/metrics';
import { createErrorResponse, validateString } from '@/lib/validation';

const logger = createLogger('api:myendpoint');

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate input
    const validation = validateString(body.name, 'name');
    if (!validation.valid) {
      return NextResponse.json(
        createErrorResponse(validation.errors[0], 'VALIDATION_ERROR'),
        { status: 400 }
      );
    }

    // Business logic...

    return NextResponse.json({ success: true, data: result });
  } catch (error) {
    logger.error('Request failed', { error });
    return NextResponse.json(
      createErrorResponse('Internal error', 'INTERNAL_ERROR'),
      { status: 500 }
    );
  }
}
```

### Adding a New Hook

1. Create hook in `src/hooks/useMyHook.ts`
2. Add `'use client'` directive for client-side hooks
3. Follow naming convention: `useFeatureName`

### Adding Contract Functionality

1. Update contract in `contracts/src/`
2. Write tests in `contracts/test/`
3. Update ABI in `src/lib/contracts.ts`
4. Add TypeScript bindings

## Deployment

### Vercel (Recommended)

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel
```

### Docker

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## Troubleshooting

### Common Issues

**"Prover unavailable" errors:**
- Check `PROVER_BACKEND_URL` is correct
- Verify prover is running (for local development)
- Mock proofs will be generated as fallback

**TypeScript errors after update:**
```bash
rm -rf node_modules .next
npm install
npm run build
```

**Contract tests failing:**
```bash
cd contracts
forge clean
forge build
forge test
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes and test
4. Commit with conventional commits: `feat: add new feature`
5. Push and create Pull Request

### Commit Message Format

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`
