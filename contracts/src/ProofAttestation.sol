// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title ProofAttestation
 * @notice Simple contract for storing zkML proof attestations on-chain
 * @dev Proof hashes are submitted by authorized attesters and can be verified by anyone
 */
contract ProofAttestation is Ownable {
    // ============ State Variables ============

    /// @notice Mapping of valid proof hashes
    mapping(bytes32 => bool) public validProofs;

    /// @notice Mapping of proof hash to submission timestamp
    mapping(bytes32 => uint256) public proofTimestamps;

    /// @notice Mapping of authorized attesters
    mapping(address => bool) public attesters;

    /// @notice Total number of attested proofs
    uint256 public totalProofs;

    // ============ Events ============

    event ProofAttested(bytes32 indexed proofHash, address indexed attester, uint256 timestamp);
    event AttesterAdded(address indexed attester);
    event AttesterRemoved(address indexed attester);

    // ============ Errors ============

    error NotAuthorized();
    error ProofAlreadyAttested();
    error InvalidProofHash();

    // ============ Constructor ============

    constructor() {
        // Owner is automatically an attester
        attesters[msg.sender] = true;
        emit AttesterAdded(msg.sender);
    }

    // ============ Modifiers ============

    modifier onlyAttester() {
        if (!attesters[msg.sender] && msg.sender != owner()) {
            revert NotAuthorized();
        }
        _;
    }

    // ============ External Functions ============

    /**
     * @notice Submit a proof attestation
     * @param proofHash The keccak256 hash of the zkML proof
     */
    function attestProof(bytes32 proofHash) external onlyAttester {
        if (proofHash == bytes32(0)) revert InvalidProofHash();
        if (validProofs[proofHash]) revert ProofAlreadyAttested();

        validProofs[proofHash] = true;
        proofTimestamps[proofHash] = block.timestamp;
        totalProofs++;

        emit ProofAttested(proofHash, msg.sender, block.timestamp);
    }

    /**
     * @notice Check if a proof hash is valid (attested)
     * @param proofHash The proof hash to check
     * @return True if the proof has been attested
     */
    function isProofHashValid(bytes32 proofHash) external view returns (bool) {
        return validProofs[proofHash];
    }

    /**
     * @notice Get the timestamp when a proof was attested
     * @param proofHash The proof hash to query
     * @return The block timestamp of attestation (0 if not attested)
     */
    function getProofTimestamp(bytes32 proofHash) external view returns (uint256) {
        return proofTimestamps[proofHash];
    }

    /**
     * @notice Add an authorized attester
     * @param attester The address to authorize
     */
    function addAttester(address attester) external onlyOwner {
        attesters[attester] = true;
        emit AttesterAdded(attester);
    }

    /**
     * @notice Remove an authorized attester
     * @param attester The address to remove
     */
    function removeAttester(address attester) external onlyOwner {
        attesters[attester] = false;
        emit AttesterRemoved(attester);
    }

    /**
     * @notice Check if an address is an authorized attester
     * @param attester The address to check
     * @return True if the address is authorized
     */
    function isAttester(address attester) external view returns (bool) {
        return attesters[attester] || attester == owner();
    }
}
