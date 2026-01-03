// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title SpendingGateWallet
 * @notice A smart contract wallet that enforces spending policy compliance via zkML proofs.
 * @dev Agents deposit USDC into this wallet. Transfers require a valid proof attestation
 *      from the ProofAttestation contract, ensuring the agent ran its policy model
 *      before spending.
 *
 * Key Features:
 * - USDC deposits from owner (the agent)
 * - Gated transfers that verify proof attestation before executing
 * - Transaction intent hashing to prevent proof reuse across different transfers
 * - Nonce management for replay protection
 * - Emergency withdrawal by owner
 */
contract SpendingGateWallet is Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    // ============ State Variables ============

    /// @notice The USDC token contract
    IERC20 public immutable usdc;

    /// @notice The ProofAttestation contract that stores valid proofs
    IProofAttestation public immutable proofAttestation;

    /// @notice Agent ID for this wallet (used in attestation lookups)
    uint256 public immutable agentId;

    /// @notice Current nonce for replay protection
    uint256 public nonce;

    /// @notice Mapping of used proof hashes (prevents double-spending with same proof)
    mapping(bytes32 => bool) public usedProofs;

    /// @notice Daily spending limit in USDC (6 decimals)
    uint256 public dailyLimit;

    /// @notice Maximum single transfer in USDC (6 decimals)
    uint256 public maxSingleTransfer;

    /// @notice Spending tracker: day => amount spent
    mapping(uint256 => uint256) public dailySpent;

    // ============ Events ============

    event Deposit(address indexed from, uint256 amount);
    event GatedTransfer(
        address indexed to,
        uint256 amount,
        bytes32 proofHash,
        bytes32 txIntentHash
    );
    event EmergencyWithdraw(address indexed to, uint256 amount);
    event LimitsUpdated(uint256 dailyLimit, uint256 maxSingleTransfer);

    // ============ Errors ============

    error InvalidProof();
    error ProofAlreadyUsed();
    error ProofNotAttested();
    error TxIntentMismatch();
    error ProofExpired();
    error ExceedsDailyLimit();
    error ExceedsMaxSingleTransfer();
    error InsufficientBalance();
    error InvalidRecipient();
    error TransferFailed();

    // ============ Constructor ============

    /**
     * @param _usdc The USDC token address
     * @param _proofAttestation The ProofAttestation contract address
     * @param _agentId The agent ID for this wallet
     * @param _dailyLimit Daily spending limit (in USDC with 6 decimals)
     * @param _maxSingleTransfer Max single transfer (in USDC with 6 decimals)
     */
    constructor(
        address _usdc,
        address _proofAttestation,
        uint256 _agentId,
        uint256 _dailyLimit,
        uint256 _maxSingleTransfer
    ) {
        usdc = IERC20(_usdc);
        proofAttestation = IProofAttestation(_proofAttestation);
        agentId = _agentId;
        dailyLimit = _dailyLimit;
        maxSingleTransfer = _maxSingleTransfer;
    }

    // ============ External Functions ============

    /**
     * @notice Deposit USDC into the wallet
     * @param amount Amount to deposit (in USDC with 6 decimals)
     */
    function deposit(uint256 amount) external {
        usdc.safeTransferFrom(msg.sender, address(this), amount);
        emit Deposit(msg.sender, amount);
    }

    /**
     * @notice Execute a gated transfer that requires a valid proof attestation
     * @param to Recipient address
     * @param amount Amount to transfer (in USDC with 6 decimals)
     * @param proofHash The hash of the zkML proof
     * @param expiry Timestamp when this transfer intent expires
     */
    function gatedTransfer(
        address to,
        uint256 amount,
        bytes32 proofHash,
        uint256 expiry
    ) external onlyOwner nonReentrant {
        // 1. Basic validation
        if (to == address(0)) revert InvalidRecipient();
        if (amount > usdc.balanceOf(address(this))) revert InsufficientBalance();
        if (block.timestamp > expiry) revert ProofExpired();

        // 2. Check spending limits
        if (amount > maxSingleTransfer) revert ExceedsMaxSingleTransfer();
        uint256 today = block.timestamp / 1 days;
        if (dailySpent[today] + amount > dailyLimit) revert ExceedsDailyLimit();

        // 3. Check proof hasn't been used
        if (usedProofs[proofHash]) revert ProofAlreadyUsed();

        // 4. Compute expected txIntentHash
        bytes32 txIntentHash = computeTxIntentHash(to, amount, nonce, expiry);

        // 5. Verify proof is attested on-chain
        if (!proofAttestation.isProofHashValid(proofHash)) {
            revert ProofNotAttested();
        }

        // 6. Mark proof as used
        usedProofs[proofHash] = true;

        // 7. Update spending tracker
        dailySpent[today] += amount;

        // 8. Increment nonce
        nonce++;

        // 9. Execute transfer
        usdc.safeTransfer(to, amount);

        emit GatedTransfer(to, amount, proofHash, txIntentHash);
    }

    /**
     * @notice Compute the transaction intent hash
     * @dev This hash binds the proof to a specific transfer intent
     */
    function computeTxIntentHash(
        address to,
        uint256 amount,
        uint256 _nonce,
        uint256 expiry
    ) public view returns (bytes32) {
        return keccak256(abi.encodePacked(
            block.chainid,
            address(usdc),
            address(this),
            to,
            amount,
            _nonce,
            expiry,
            agentId
        ));
    }

    /**
     * @notice Get current nonce for building transaction intent
     */
    function getCurrentNonce() external view returns (uint256) {
        return nonce;
    }

    /**
     * @notice Get remaining daily allowance
     */
    function getRemainingDailyAllowance() external view returns (uint256) {
        uint256 today = block.timestamp / 1 days;
        uint256 spent = dailySpent[today];
        if (spent >= dailyLimit) return 0;
        return dailyLimit - spent;
    }

    /**
     * @notice Check if a proof has been used
     */
    function isProofUsed(bytes32 proofHash) external view returns (bool) {
        return usedProofs[proofHash];
    }

    /**
     * @notice Update spending limits (owner only)
     */
    function updateLimits(
        uint256 _dailyLimit,
        uint256 _maxSingleTransfer
    ) external onlyOwner {
        dailyLimit = _dailyLimit;
        maxSingleTransfer = _maxSingleTransfer;
        emit LimitsUpdated(_dailyLimit, _maxSingleTransfer);
    }

    /**
     * @notice Emergency withdrawal (owner only)
     * @dev Use only in emergencies - bypasses proof requirement
     */
    function emergencyWithdraw(address to) external onlyOwner {
        uint256 balance = usdc.balanceOf(address(this));
        if (balance > 0) {
            usdc.safeTransfer(to, balance);
            emit EmergencyWithdraw(to, balance);
        }
    }

    /**
     * @notice Get wallet balance
     */
    function getBalance() external view returns (uint256) {
        return usdc.balanceOf(address(this));
    }
}

/**
 * @title IProofAttestation
 * @notice Interface for the ProofAttestation contract
 */
interface IProofAttestation {
    function isProofHashValid(bytes32 proofHash) external view returns (bool);

    function getProofMetadata(bytes32 requestHash) external view returns (
        bytes32 modelHash,
        bytes32 inputHash,
        bytes32 outputHash,
        uint256 proofSize,
        uint256 generationTime,
        string memory proverVersion
    );
}
