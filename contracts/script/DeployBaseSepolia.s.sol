// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "forge-std/console.sol";
import "../src/ProofAttestation.sol";
import "../src/SpendingGateWallet.sol";

/**
 * @title DeployBaseSepolia
 * @notice Deployment script for ProofAttestation and SpendingGateWallet on Base Sepolia
 * @dev Uses Crossmint's testnet USDC at 0x8a04d904055528a69f3e4594dda308a31aeb8457
 */
contract DeployBaseSepolia is Script {
    // Crossmint's testnet USDC on Base Sepolia
    address constant USDC = 0x8A04d904055528a69f3E4594DDA308A31aeb8457;

    // Default agent ID
    uint256 constant DEFAULT_AGENT_ID = 1;

    // Default limits (in USDC with 6 decimals)
    uint256 constant DAILY_LIMIT = 10000 * 1e6; // 10,000 USDC
    uint256 constant MAX_SINGLE_TRANSFER = 5000 * 1e6; // 5,000 USDC

    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerPrivateKey);

        console.log("=== Base Sepolia Deployment ===");
        console.log("Deployer address:", deployer);
        console.log("Deployer balance:", deployer.balance);

        vm.startBroadcast(deployerPrivateKey);

        // 1. Deploy ProofAttestation
        ProofAttestation proofAttestation = new ProofAttestation();
        console.log("ProofAttestation deployed at:", address(proofAttestation));

        // 2. Deploy SpendingGateWallet
        SpendingGateWallet wallet = new SpendingGateWallet(
            USDC,
            address(proofAttestation),
            DEFAULT_AGENT_ID,
            DAILY_LIMIT,
            MAX_SINGLE_TRANSFER
        );
        console.log("SpendingGateWallet deployed at:", address(wallet));

        // 3. Add the Crossmint wallet as an attester (optional)
        // proofAttestation.addAttester(CROSSMINT_WALLET);

        vm.stopBroadcast();

        // Output for .env update
        console.log("\n=== Add to .env.local ===");
        console.log("NEXT_PUBLIC_BASE_SEPOLIA_PROOF_ATTESTATION=", address(proofAttestation));
        console.log("NEXT_PUBLIC_BASE_SEPOLIA_SPENDING_GATE=", address(wallet));
        console.log("NEXT_PUBLIC_BASE_SEPOLIA_USDC=", USDC);
    }
}
