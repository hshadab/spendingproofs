// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title MockUSDC
 * @notice A mock USDC token for Arc Testnet testing
 * @dev Anyone can mint for testing purposes
 */
contract MockUSDC is ERC20, Ownable {
    uint8 private constant _DECIMALS = 6;

    constructor() ERC20("USD Coin", "USDC") {
        // Mint 1 million USDC to deployer for initial testing
        _mint(msg.sender, 1_000_000 * 10 ** _DECIMALS);
    }

    function decimals() public pure override returns (uint8) {
        return _DECIMALS;
    }

    /**
     * @notice Mint tokens (for testing only)
     */
    function mint(address to, uint256 amount) external {
        _mint(to, amount);
    }

    /**
     * @notice Faucet - get 100 USDC for testing
     */
    function faucet() external {
        _mint(msg.sender, 100 * 10 ** _DECIMALS);
    }
}
