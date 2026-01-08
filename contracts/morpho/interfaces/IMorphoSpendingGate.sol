// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IMorphoSpendingGate
 * @notice Interface for zkML-gated Morpho Blue operations
 * @dev Agents must submit Jolt-Atlas SNARK proofs to execute vault operations
 */
interface IMorphoSpendingGate {
    /// @notice Spending policy configuration
    struct SpendingPolicy {
        uint256 dailyLimit;           // Maximum daily spend in base units
        uint256 maxSingleTx;          // Maximum single transaction amount
        uint256 maxLTV;               // Maximum loan-to-value ratio (basis points, e.g., 7000 = 70%)
        uint256 minHealthFactor;      // Minimum health factor (basis points, e.g., 12000 = 1.2)
        address[] allowedMarkets;     // Whitelisted Morpho markets
        bool requireProofForSupply;   // Require proof for supply operations
        bool requireProofForBorrow;   // Require proof for borrow operations
        bool requireProofForWithdraw; // Require proof for withdraw operations
    }

    /// @notice Proof submission data
    struct SpendingProof {
        bytes32 policyHash;           // Hash of the policy being proven against
        bytes proof;                  // Jolt-Atlas SNARK proof (~48KB)
        bytes32[] publicInputs;       // Public inputs to the proof
        uint256 timestamp;            // Proof generation timestamp
        bytes signature;              // Agent signature over proof commitment
    }

    /// @notice Agent registration data
    struct AgentConfig {
        address agent;                // Agent address
        address owner;                // Vault owner who authorized the agent
        bytes32 policyHash;           // Active policy hash
        uint256 dailySpent;           // Amount spent today
        uint256 lastResetTimestamp;   // Last daily reset timestamp
        bool isActive;                // Whether agent is currently authorized
    }

    // Events
    event PolicyRegistered(address indexed owner, bytes32 indexed policyHash, SpendingPolicy policy);
    event AgentAuthorized(address indexed owner, address indexed agent, bytes32 policyHash);
    event AgentRevoked(address indexed owner, address indexed agent);
    event ProofVerified(address indexed agent, bytes32 indexed policyHash, bytes32 proofHash);
    event SupplyExecuted(address indexed agent, address indexed market, uint256 amount, bytes32 proofHash);
    event BorrowExecuted(address indexed agent, address indexed market, uint256 amount, bytes32 proofHash);
    event WithdrawExecuted(address indexed agent, address indexed market, uint256 amount, bytes32 proofHash);
    event RepayExecuted(address indexed agent, address indexed market, uint256 amount, bytes32 proofHash);

    // Errors
    error InvalidProof();
    error PolicyNotFound();
    error AgentNotAuthorized();
    error ExceedsDailyLimit();
    error ExceedsSingleTxLimit();
    error MarketNotAllowed();
    error LTVExceeded();
    error HealthFactorTooLow();
    error ProofExpired();
    error InvalidSignature();

    // Policy Management
    function registerPolicy(SpendingPolicy calldata policy) external returns (bytes32 policyHash);
    function getPolicy(bytes32 policyHash) external view returns (SpendingPolicy memory);
    function getPolicyHash(SpendingPolicy calldata policy) external pure returns (bytes32);

    // Agent Management
    function authorizeAgent(address agent, bytes32 policyHash) external;
    function revokeAgent(address agent) external;
    function getAgentConfig(address agent) external view returns (AgentConfig memory);
    function isAgentAuthorized(address agent, address owner) external view returns (bool);

    // Gated Operations (require zkML proof)
    function supplyWithProof(
        address market,
        uint256 assets,
        address onBehalf,
        SpendingProof calldata proof
    ) external returns (uint256 shares);

    function borrowWithProof(
        address market,
        uint256 assets,
        address onBehalf,
        address receiver,
        SpendingProof calldata proof
    ) external returns (uint256 shares);

    function withdrawWithProof(
        address market,
        uint256 assets,
        address onBehalf,
        address receiver,
        SpendingProof calldata proof
    ) external returns (uint256 shares);

    function repayWithProof(
        address market,
        uint256 assets,
        address onBehalf,
        SpendingProof calldata proof
    ) external returns (uint256 repaid);

    // View Functions
    function getDailySpent(address agent) external view returns (uint256);
    function getRemainingDailyLimit(address agent) external view returns (uint256);
    function verifyProof(SpendingProof calldata proof) external view returns (bool);
}
