// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IJoltAtlasVerifier.sol";

/**
 * @title MockJoltAtlasVerifier
 * @notice Mock verifier for testing - accepts all proofs
 * @dev In production, this would perform actual Jolt-Atlas SNARK verification
 */
contract MockJoltAtlasVerifier is IJoltAtlasVerifier {
    mapping(string => VerificationKey) private verificationKeys;
    mapping(string => bool) private keyExists;

    bool public alwaysVerify = true;

    function setAlwaysVerify(bool _value) external {
        alwaysVerify = _value;
    }

    function verify(
        bytes calldata proof,
        bytes32[] calldata publicInputs,
        bytes32 policyHash
    ) external view override returns (bool valid) {
        // Basic validation
        if (proof.length < 256) return false;
        if (publicInputs.length < 7) return false;
        if (policyHash == bytes32(0)) return false;

        return alwaysVerify;
    }

    function registerVerificationKey(
        string calldata policyType,
        VerificationKey calldata vk
    ) external override {
        verificationKeys[policyType] = vk;
        keyExists[policyType] = true;
    }

    function getVerificationKey(
        string calldata policyType
    ) external view override returns (VerificationKey memory vk) {
        return verificationKeys[policyType];
    }

    function hasVerificationKey(
        string calldata policyType
    ) external view override returns (bool exists) {
        return keyExists[policyType];
    }
}
