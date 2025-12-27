/**
 * Mock PolicyRegistry Contract Interface
 *
 * The PolicyRegistry prevents model substitution attacks by maintaining
 * an on-chain mapping of approved policies to their model and VK hashes.
 *
 * Without a registry, a malicious agent could:
 * 1. Create a permissive model that always approves
 * 2. Generate a valid proof using that model
 * 3. Present it as if it came from the legitimate policy
 *
 * The registry ensures verifiers can check:
 * - Is this policyId recognized?
 * - Does the modelHash match what's registered?
 * - Does the vkHash match the expected verification key?
 */

export interface PolicyInfo {
  policyId: string;
  modelHash: string;
  vkHash: string;
  metadataURI: string;
  version: number;
  registeredAt: number;
  owner: string;
  isActive: boolean;
}

export interface RegistryLookupResult {
  found: boolean;
  policy?: PolicyInfo;
  error?: string;
}

export interface PolicyValidationResult {
  valid: boolean;
  policyId: string;
  modelHashMatch: boolean;
  vkHashMatch: boolean;
  isActive: boolean;
  versionMatch: boolean;
  expectedVersion?: number;
  reason?: string;
}

// Mock registry storage
const policyRegistry: Map<string, PolicyInfo> = new Map([
  ['default-spending-policy', {
    policyId: 'default-spending-policy',
    modelHash: '0x7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b3c4d5e6f7a8b',
    vkHash: '0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
    metadataURI: 'ipfs://QmSpendingPolicyV1',
    version: 1,
    registeredAt: Date.now() - 86400000, // 1 day ago
    owner: '0x742d35Cc6634C0532925a3b844Bc9e7595f8fE32',
    isActive: true,
  }],
  ['enterprise-spending-policy', {
    policyId: 'enterprise-spending-policy',
    modelHash: '0xabcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890',
    vkHash: '0xfedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321',
    metadataURI: 'ipfs://QmEnterprisePolicyV2',
    version: 2,
    registeredAt: Date.now() - 43200000, // 12 hours ago
    owner: '0x8ba1f109551bD432803012645Ac136ddd64DBA72',
    isActive: true,
  }],
  ['deprecated-policy', {
    policyId: 'deprecated-policy',
    modelHash: '0x0000111122223333444455556666777788889999aaaabbbbccccddddeeeeffff',
    vkHash: '0xffffeeeeddddccccbbbbaaaa99998888777766665555444433332222111100000',
    metadataURI: 'ipfs://QmDeprecatedPolicy',
    version: 0,
    registeredAt: Date.now() - 604800000, // 1 week ago
    owner: '0x742d35Cc6634C0532925a3b844Bc9e7595f8fE32',
    isActive: false,
  }],
]);

/**
 * Look up a policy by ID
 */
export async function lookupPolicy(policyId: string): Promise<RegistryLookupResult> {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 200));

  const policy = policyRegistry.get(policyId);

  if (!policy) {
    return {
      found: false,
      error: `Policy "${policyId}" not found in registry`,
    };
  }

  return {
    found: true,
    policy,
  };
}

/**
 * Validate a proof's policy against the registry
 */
export async function validatePolicyProof(
  policyId: string,
  modelHash: string,
  vkHash: string,
  version?: number
): Promise<PolicyValidationResult> {
  // Simulate network delay
  await new Promise(resolve => setTimeout(resolve, 300));

  const policy = policyRegistry.get(policyId);

  if (!policy) {
    return {
      valid: false,
      policyId,
      modelHashMatch: false,
      vkHashMatch: false,
      isActive: false,
      versionMatch: false,
      reason: 'UNKNOWN_POLICY - Policy ID not found in registry',
    };
  }

  const modelHashMatch = policy.modelHash.toLowerCase() === modelHash.toLowerCase();
  const vkHashMatch = policy.vkHash.toLowerCase() === vkHash.toLowerCase();
  const versionMatch = version === undefined || policy.version === version;

  if (!policy.isActive) {
    return {
      valid: false,
      policyId,
      modelHashMatch,
      vkHashMatch,
      isActive: false,
      versionMatch,
      expectedVersion: policy.version,
      reason: 'POLICY_INACTIVE - Policy has been deprecated',
    };
  }

  if (!modelHashMatch) {
    return {
      valid: false,
      policyId,
      modelHashMatch,
      vkHashMatch,
      isActive: true,
      versionMatch,
      expectedVersion: policy.version,
      reason: 'MODEL_MISMATCH - Proof model hash does not match registered model',
    };
  }

  if (!vkHashMatch) {
    return {
      valid: false,
      policyId,
      modelHashMatch,
      vkHashMatch,
      isActive: true,
      versionMatch,
      expectedVersion: policy.version,
      reason: 'VK_MISMATCH - Verification key hash does not match registered VK',
    };
  }

  if (!versionMatch) {
    return {
      valid: false,
      policyId,
      modelHashMatch,
      vkHashMatch,
      isActive: true,
      versionMatch,
      expectedVersion: policy.version,
      reason: `VERSION_MISMATCH - Expected v${policy.version}, got v${version}`,
    };
  }

  return {
    valid: true,
    policyId,
    modelHashMatch: true,
    vkHashMatch: true,
    isActive: true,
    versionMatch: true,
    expectedVersion: policy.version,
  };
}

/**
 * Get all registered policies
 */
export async function getAllPolicies(): Promise<PolicyInfo[]> {
  await new Promise(resolve => setTimeout(resolve, 100));
  return Array.from(policyRegistry.values());
}

/**
 * Register a new policy (mock)
 */
export async function registerPolicy(policy: Omit<PolicyInfo, 'registeredAt'>): Promise<boolean> {
  await new Promise(resolve => setTimeout(resolve, 500));

  policyRegistry.set(policy.policyId, {
    ...policy,
    registeredAt: Date.now(),
  });

  return true;
}

/**
 * Check if a model hash is known in any registered policy
 */
export async function isKnownModel(modelHash: string): Promise<{ known: boolean; policyId?: string }> {
  await new Promise(resolve => setTimeout(resolve, 150));

  for (const [policyId, policy] of policyRegistry) {
    if (policy.modelHash.toLowerCase() === modelHash.toLowerCase()) {
      return { known: true, policyId };
    }
  }

  return { known: false };
}
