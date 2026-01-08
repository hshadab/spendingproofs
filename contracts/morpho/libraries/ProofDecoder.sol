// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title ProofDecoder
 * @notice Library for decoding Jolt-Atlas SNARK proofs
 * @dev Handles the ~48KB proof format from Jolt-Atlas prover
 */
library ProofDecoder {
    /// @notice Decoded proof structure
    struct DecodedProof {
        uint256[2] a;           // G1 point
        uint256[2][2] b;        // G2 point
        uint256[2] c;           // G1 point
        uint256[] publicSignals;
    }

    /// @notice Proof metadata extracted from public inputs
    struct ProofMetadata {
        bytes32 policyHash;
        uint256 operationType;  // 0: supply, 1: borrow, 2: withdraw, 3: repay
        uint256 amount;
        address market;
        address agent;
        uint256 timestamp;
        uint256 nonce;
    }

    uint256 constant OPERATION_SUPPLY = 0;
    uint256 constant OPERATION_BORROW = 1;
    uint256 constant OPERATION_WITHDRAW = 2;
    uint256 constant OPERATION_REPAY = 3;

    /// @notice Decode a Jolt-Atlas proof
    function decode(bytes calldata proof) internal pure returns (DecodedProof memory decoded) {
        require(proof.length >= 256, "Proof too short");

        // Decode G1 point a (64 bytes)
        decoded.a[0] = abi.decode(proof[0:32], (uint256));
        decoded.a[1] = abi.decode(proof[32:64], (uint256));

        // Decode G2 point b (128 bytes)
        decoded.b[0][0] = abi.decode(proof[64:96], (uint256));
        decoded.b[0][1] = abi.decode(proof[96:128], (uint256));
        decoded.b[1][0] = abi.decode(proof[128:160], (uint256));
        decoded.b[1][1] = abi.decode(proof[160:192], (uint256));

        // Decode G1 point c (64 bytes)
        decoded.c[0] = abi.decode(proof[192:224], (uint256));
        decoded.c[1] = abi.decode(proof[224:256], (uint256));

        // Remaining bytes are public signals
        if (proof.length > 256) {
            uint256 numSignals = (proof.length - 256) / 32;
            decoded.publicSignals = new uint256[](numSignals);
            for (uint256 i = 0; i < numSignals; i++) {
                decoded.publicSignals[i] = abi.decode(
                    proof[256 + i * 32:288 + i * 32],
                    (uint256)
                );
            }
        }
    }

    /// @notice Extract metadata from public inputs
    function extractMetadata(
        bytes32[] calldata publicInputs
    ) internal pure returns (ProofMetadata memory metadata) {
        require(publicInputs.length >= 7, "Insufficient public inputs");

        metadata.policyHash = publicInputs[0];
        metadata.operationType = uint256(publicInputs[1]);
        metadata.amount = uint256(publicInputs[2]);
        metadata.market = address(uint160(uint256(publicInputs[3])));
        metadata.agent = address(uint160(uint256(publicInputs[4])));
        metadata.timestamp = uint256(publicInputs[5]);
        metadata.nonce = uint256(publicInputs[6]);
    }

    /// @notice Compute proof hash for deduplication/caching
    function computeProofHash(bytes calldata proof) internal pure returns (bytes32) {
        return keccak256(proof);
    }

    /// @notice Validate proof structure
    function validateStructure(bytes calldata proof) internal pure returns (bool) {
        // Minimum size: 256 bytes for curve points + at least 224 bytes for 7 public inputs
        if (proof.length < 480) return false;

        // Check alignment
        if ((proof.length - 256) % 32 != 0) return false;

        return true;
    }

    /// @notice Check if operation type is valid
    function isValidOperation(uint256 operationType) internal pure returns (bool) {
        return operationType <= OPERATION_REPAY;
    }
}
