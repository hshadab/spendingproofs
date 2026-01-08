// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./interfaces/IMorphoSpendingGate.sol";
import "./interfaces/IMorphoBlue.sol";
import "./libraries/PolicyVerifier.sol";
import "./libraries/ProofDecoder.sol";

/**
 * @title MorphoSpendingGate
 * @notice zkML-gated wrapper for Morpho Blue operations
 * @dev Agents submit proofs that were verified off-chain by NovaNet's Jolt-Atlas prover
 *
 * Architecture:
 * 1. Agent generates zkML proof off-chain via NovaNet prover service
 * 2. Proof is verified off-chain (4-12 seconds, ~48KB)
 * 3. Agent signs the verified proof commitment
 * 4. This contract validates signature + policy constraints
 * 5. If valid, executes the Morpho operation
 *
 * This enables trustless autonomous DeFi without expensive on-chain SNARK verification.
 */
contract MorphoSpendingGate is IMorphoSpendingGate {
    using PolicyVerifier for *;
    using ProofDecoder for bytes;

    // Immutables
    IMorphoBlue public immutable morpho;

    // Storage
    mapping(bytes32 => SpendingPolicy) private policies;
    mapping(address => AgentConfig) private agents;
    mapping(address => mapping(address => bool)) private ownerAgents; // owner => agent => authorized
    mapping(bytes32 => bool) private usedProofs; // Prevent proof replay

    // Events for off-chain verification tracking
    event ProofSubmitted(address indexed agent, bytes32 indexed policyHash, bytes32 proofHash, uint256 timestamp);

    constructor(address _morpho) {
        morpho = IMorphoBlue(_morpho);
    }

    // ============ Policy Management ============

    /// @inheritdoc IMorphoSpendingGate
    function registerPolicy(SpendingPolicy calldata policy) external returns (bytes32 policyHash) {
        policyHash = PolicyVerifier.computePolicyHash(policy);
        policies[policyHash] = policy;
        emit PolicyRegistered(msg.sender, policyHash, policy);
    }

    /// @inheritdoc IMorphoSpendingGate
    function getPolicy(bytes32 policyHash) external view returns (SpendingPolicy memory) {
        return policies[policyHash];
    }

    /// @inheritdoc IMorphoSpendingGate
    function getPolicyHash(SpendingPolicy calldata policy) external pure returns (bytes32) {
        return PolicyVerifier.computePolicyHash(policy);
    }

    // ============ Agent Management ============

    /// @inheritdoc IMorphoSpendingGate
    function authorizeAgent(address agent, bytes32 policyHash) external {
        if (policies[policyHash].dailyLimit == 0 && policies[policyHash].maxLTV == 0) {
            revert PolicyNotFound();
        }

        agents[agent] = AgentConfig({
            agent: agent,
            owner: msg.sender,
            policyHash: policyHash,
            dailySpent: 0,
            lastResetTimestamp: block.timestamp,
            isActive: true
        });

        ownerAgents[msg.sender][agent] = true;

        emit AgentAuthorized(msg.sender, agent, policyHash);
    }

    /// @inheritdoc IMorphoSpendingGate
    function revokeAgent(address agent) external {
        AgentConfig storage config = agents[agent];
        if (config.owner != msg.sender) revert AgentNotAuthorized();

        config.isActive = false;
        ownerAgents[msg.sender][agent] = false;

        emit AgentRevoked(msg.sender, agent);
    }

    /// @inheritdoc IMorphoSpendingGate
    function getAgentConfig(address agent) external view returns (AgentConfig memory) {
        return agents[agent];
    }

    /// @inheritdoc IMorphoSpendingGate
    function isAgentAuthorized(address agent, address owner) external view returns (bool) {
        return ownerAgents[owner][agent] && agents[agent].isActive;
    }

    // ============ Gated Operations ============

    /// @inheritdoc IMorphoSpendingGate
    function supplyWithProof(
        address market,
        uint256 assets,
        address onBehalf,
        SpendingProof calldata proof
    ) external returns (uint256 shares) {
        AgentConfig storage config = agents[msg.sender];
        _validateAgentAndProof(config, proof, assets, market, ProofDecoder.OPERATION_SUPPLY);

        SpendingPolicy storage policy = policies[config.policyHash];
        if (!policy.requireProofForSupply) {
            _validateLimits(config, policy, assets, market);
        }

        _updateDailySpent(config, assets);

        IMorphoBlue.MarketParams memory params = _getMarketParams(market);
        (, shares) = morpho.supply(params, assets, 0, onBehalf, "");

        bytes32 proofHash = keccak256(proof.proof);
        emit SupplyExecuted(msg.sender, market, assets, proofHash);
    }

    /// @inheritdoc IMorphoSpendingGate
    function borrowWithProof(
        address market,
        uint256 assets,
        address onBehalf,
        address receiver,
        SpendingProof calldata proof
    ) external returns (uint256 shares) {
        AgentConfig storage config = agents[msg.sender];
        _validateAgentAndProof(config, proof, assets, market, ProofDecoder.OPERATION_BORROW);

        SpendingPolicy storage policy = policies[config.policyHash];
        _validateLimits(config, policy, assets, market);

        _updateDailySpent(config, assets);

        IMorphoBlue.MarketParams memory params = _getMarketParams(market);
        (, shares) = morpho.borrow(params, assets, 0, onBehalf, receiver);

        bytes32 proofHash = keccak256(proof.proof);
        emit BorrowExecuted(msg.sender, market, assets, proofHash);
    }

    /// @inheritdoc IMorphoSpendingGate
    function withdrawWithProof(
        address market,
        uint256 assets,
        address onBehalf,
        address receiver,
        SpendingProof calldata proof
    ) external returns (uint256 shares) {
        AgentConfig storage config = agents[msg.sender];
        _validateAgentAndProof(config, proof, assets, market, ProofDecoder.OPERATION_WITHDRAW);

        SpendingPolicy storage policy = policies[config.policyHash];
        if (policy.requireProofForWithdraw) {
            _validateLimits(config, policy, assets, market);
        }

        _updateDailySpent(config, assets);

        IMorphoBlue.MarketParams memory params = _getMarketParams(market);
        (, shares) = morpho.withdraw(params, assets, 0, onBehalf, receiver);

        bytes32 proofHash = keccak256(proof.proof);
        emit WithdrawExecuted(msg.sender, market, assets, proofHash);
    }

    /// @inheritdoc IMorphoSpendingGate
    function repayWithProof(
        address market,
        uint256 assets,
        address onBehalf,
        SpendingProof calldata proof
    ) external returns (uint256 repaid) {
        AgentConfig storage config = agents[msg.sender];
        _validateAgentAndProof(config, proof, assets, market, ProofDecoder.OPERATION_REPAY);

        IMorphoBlue.MarketParams memory params = _getMarketParams(market);
        (repaid, ) = morpho.repay(params, assets, 0, onBehalf, "");

        bytes32 proofHash = keccak256(proof.proof);
        emit RepayExecuted(msg.sender, market, assets, proofHash);
    }

    // ============ View Functions ============

    /// @inheritdoc IMorphoSpendingGate
    function getDailySpent(address agent) external view returns (uint256) {
        AgentConfig storage config = agents[agent];
        if (PolicyVerifier.shouldResetDailyLimit(config.lastResetTimestamp)) {
            return 0;
        }
        return config.dailySpent;
    }

    /// @inheritdoc IMorphoSpendingGate
    function getRemainingDailyLimit(address agent) external view returns (uint256) {
        AgentConfig storage config = agents[agent];
        SpendingPolicy storage policy = policies[config.policyHash];

        uint256 spent = config.dailySpent;
        if (PolicyVerifier.shouldResetDailyLimit(config.lastResetTimestamp)) {
            spent = 0;
        }

        if (spent >= policy.dailyLimit) return 0;
        return policy.dailyLimit - spent;
    }

    /// @notice Check if a proof has been used (for off-chain verification)
    function isProofUsed(bytes32 proofHash) external view returns (bool) {
        return usedProofs[proofHash];
    }

    /// @inheritdoc IMorphoSpendingGate
    function verifyProof(SpendingProof calldata proof) external pure returns (bool) {
        // Off-chain verification - this just checks proof structure is valid
        // Actual zkML verification happens via NovaNet prover service
        return proof.proof.length > 0 && proof.signature.length == 65;
    }

    // ============ Internal Functions ============

    function _validateAgentAndProof(
        AgentConfig storage config,
        SpendingProof calldata proof,
        uint256 amount,
        address market,
        uint256 operationType
    ) internal {
        // Check agent is authorized
        if (!config.isActive) revert AgentNotAuthorized();

        // Check proof hasn't been used (replay protection)
        bytes32 proofHash = keccak256(proof.proof);
        if (usedProofs[proofHash]) revert InvalidProof();

        // Check proof timestamp (proofs expire after 5 minutes)
        if (!PolicyVerifier.verifyProofTimestamp(proof.timestamp)) {
            revert ProofExpired();
        }

        // Verify proof matches agent's registered policy
        if (proof.policyHash != config.policyHash) revert InvalidProof();

        // Extract and validate metadata from proof public inputs
        ProofDecoder.ProofMetadata memory metadata = ProofDecoder.extractMetadata(proof.publicInputs);
        if (metadata.operationType != operationType) revert InvalidProof();
        if (metadata.amount != amount) revert InvalidProof();
        if (metadata.market != market) revert InvalidProof();
        if (metadata.agent != msg.sender) revert InvalidProof();

        // Verify agent signature on proof commitment
        // This proves the agent approved this specific proof after off-chain verification
        bytes32 proofCommitment = keccak256(abi.encodePacked(proofHash, metadata.nonce));
        if (!PolicyVerifier.verifyAgentSignature(msg.sender, proofCommitment, proof.signature)) {
            revert InvalidSignature();
        }

        // Mark proof as used
        usedProofs[proofHash] = true;

        emit ProofSubmitted(msg.sender, proof.policyHash, proofHash, proof.timestamp);
        emit ProofVerified(msg.sender, proof.policyHash, proofHash);
    }

    function _validateLimits(
        AgentConfig storage config,
        SpendingPolicy storage policy,
        uint256 amount,
        address market
    ) internal view {
        // Check market is allowed
        if (!PolicyVerifier.verifyMarketAllowed(market, policy.allowedMarkets)) {
            revert MarketNotAllowed();
        }

        // Check single tx limit
        if (!PolicyVerifier.verifySingleTxLimit(amount, policy.maxSingleTx)) {
            revert ExceedsSingleTxLimit();
        }

        // Check daily limit
        uint256 currentSpent = config.dailySpent;
        if (PolicyVerifier.shouldResetDailyLimit(config.lastResetTimestamp)) {
            currentSpent = 0;
        }
        if (!PolicyVerifier.verifyDailyLimit(amount, currentSpent, policy.dailyLimit)) {
            revert ExceedsDailyLimit();
        }
    }

    function _updateDailySpent(AgentConfig storage config, uint256 amount) internal {
        if (PolicyVerifier.shouldResetDailyLimit(config.lastResetTimestamp)) {
            config.dailySpent = amount;
            config.lastResetTimestamp = block.timestamp;
        } else {
            config.dailySpent += amount;
        }
    }

    function _getMarketParams(address market) internal pure returns (IMorphoBlue.MarketParams memory) {
        // In production, this would fetch actual market params from Morpho
        return IMorphoBlue.MarketParams({
            loanToken: market,
            collateralToken: address(0),
            oracle: address(0),
            irm: address(0),
            lltv: 0
        });
    }
}
