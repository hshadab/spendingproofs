// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "forge-std/Script.sol";
import "forge-std/console.sol";
import "../src/SpendingGateWallet.sol";
import "../src/MockUSDC.sol";

/**
 * @title Deploy
 * @notice Deployment script for SpendingGateWallet and MockUSDC on Arc Testnet
 */
contract Deploy is Script {
    // Arc Testnet ProofAttestation contract
    address constant PROOF_ATTESTATION = 0xBE9a5DF7C551324CB872584C6E5bF56799787952;

    // Default agent ID (can be overridden)
    uint256 constant DEFAULT_AGENT_ID = 1;

    // Default limits (in USDC with 6 decimals)
    uint256 constant DAILY_LIMIT = 1000 * 1e6; // 1000 USDC
    uint256 constant MAX_SINGLE_TRANSFER = 100 * 1e6; // 100 USDC

    function run() external {
        uint256 deployerPrivateKey = vm.envUint("PRIVATE_KEY");
        address deployer = vm.addr(deployerPrivateKey);

        console.log("Deployer address:", deployer);
        console.log("Deployer balance:", deployer.balance);

        vm.startBroadcast(deployerPrivateKey);

        // 1. Deploy MockUSDC
        MockUSDC usdc = new MockUSDC();
        console.log("MockUSDC deployed at:", address(usdc));
        console.log("Deployer USDC balance:", usdc.balanceOf(deployer));

        // 2. Deploy SpendingGateWallet
        SpendingGateWallet wallet = new SpendingGateWallet(
            address(usdc),
            PROOF_ATTESTATION,
            DEFAULT_AGENT_ID,
            DAILY_LIMIT,
            MAX_SINGLE_TRANSFER
        );
        console.log("SpendingGateWallet deployed at:", address(wallet));

        // 3. Approve and deposit 100 USDC into the wallet
        uint256 depositAmount = 100 * 1e6; // 100 USDC
        usdc.approve(address(wallet), depositAmount);
        wallet.deposit(depositAmount);
        console.log("Deposited", depositAmount / 1e6, "USDC into SpendingGateWallet");
        console.log("Wallet USDC balance:", wallet.getBalance() / 1e6, "USDC");

        vm.stopBroadcast();

        // Output for .env update
        console.log("\n=== Add to .env.local ===");
        console.log("NEXT_PUBLIC_USDC_ADDRESS=", address(usdc));
        console.log("NEXT_PUBLIC_SPENDING_GATE_ADDRESS=", address(wallet));
    }
}
