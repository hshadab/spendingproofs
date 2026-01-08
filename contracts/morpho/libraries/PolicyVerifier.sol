// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IMorphoSpendingGate.sol";

/**
 * @title PolicyVerifier
 * @notice Library for verifying spending policy constraints
 * @dev Used by MorphoSpendingGate to validate operations against policies
 */
library PolicyVerifier {
    uint256 constant BASIS_POINTS = 10000;
    uint256 constant DAY_IN_SECONDS = 86400;
    uint256 constant PROOF_VALIDITY_WINDOW = 300; // 5 minutes

    /// @notice Verify that an amount doesn't exceed daily limit
    function verifyDailyLimit(
        uint256 amount,
        uint256 dailySpent,
        uint256 dailyLimit
    ) internal pure returns (bool) {
        return (dailySpent + amount) <= dailyLimit;
    }

    /// @notice Verify that an amount doesn't exceed single tx limit
    function verifySingleTxLimit(
        uint256 amount,
        uint256 maxSingleTx
    ) internal pure returns (bool) {
        if (maxSingleTx == 0) return true; // No limit set
        return amount <= maxSingleTx;
    }

    /// @notice Verify that a market is in the allowed list
    function verifyMarketAllowed(
        address market,
        address[] memory allowedMarkets
    ) internal pure returns (bool) {
        if (allowedMarkets.length == 0) return true; // No restrictions

        for (uint256 i = 0; i < allowedMarkets.length; i++) {
            if (allowedMarkets[i] == market) {
                return true;
            }
        }
        return false;
    }

    /// @notice Verify LTV is within bounds
    /// @param borrowValue Total borrow value in base units
    /// @param collateralValue Total collateral value in base units
    /// @param maxLTV Maximum LTV in basis points (e.g., 7000 = 70%)
    function verifyLTV(
        uint256 borrowValue,
        uint256 collateralValue,
        uint256 maxLTV
    ) internal pure returns (bool) {
        if (collateralValue == 0) {
            return borrowValue == 0;
        }
        uint256 currentLTV = (borrowValue * BASIS_POINTS) / collateralValue;
        return currentLTV <= maxLTV;
    }

    /// @notice Verify health factor is above minimum
    /// @param healthFactor Current health factor in basis points
    /// @param minHealthFactor Minimum required health factor
    function verifyHealthFactor(
        uint256 healthFactor,
        uint256 minHealthFactor
    ) internal pure returns (bool) {
        return healthFactor >= minHealthFactor;
    }

    /// @notice Check if daily limit should be reset
    function shouldResetDailyLimit(
        uint256 lastResetTimestamp
    ) internal view returns (bool) {
        return block.timestamp >= lastResetTimestamp + DAY_IN_SECONDS;
    }

    /// @notice Verify proof timestamp is within validity window
    function verifyProofTimestamp(
        uint256 proofTimestamp
    ) internal view returns (bool) {
        return block.timestamp <= proofTimestamp + PROOF_VALIDITY_WINDOW;
    }

    /// @notice Compute policy hash
    function computePolicyHash(
        IMorphoSpendingGate.SpendingPolicy memory policy
    ) internal pure returns (bytes32) {
        return keccak256(abi.encode(
            policy.dailyLimit,
            policy.maxSingleTx,
            policy.maxLTV,
            policy.minHealthFactor,
            policy.allowedMarkets,
            policy.requireProofForSupply,
            policy.requireProofForBorrow,
            policy.requireProofForWithdraw
        ));
    }

    /// @notice Verify agent signature over proof commitment
    function verifyAgentSignature(
        address agent,
        bytes32 proofCommitment,
        bytes memory signature
    ) internal pure returns (bool) {
        bytes32 ethSignedHash = keccak256(abi.encodePacked(
            "\x19Ethereum Signed Message:\n32",
            proofCommitment
        ));

        (bytes32 r, bytes32 s, uint8 v) = splitSignature(signature);
        address recovered = ecrecover(ethSignedHash, v, r, s);

        return recovered == agent;
    }

    /// @notice Split signature into r, s, v components
    function splitSignature(bytes memory sig)
        internal
        pure
        returns (bytes32 r, bytes32 s, uint8 v)
    {
        require(sig.length == 65, "Invalid signature length");

        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }

        if (v < 27) {
            v += 27;
        }
    }
}
