// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IMorphoBlue
 * @notice Minimal interface for Morpho Blue protocol interactions
 * @dev Based on Morpho Blue's actual interface
 */
interface IMorphoBlue {
    /// @notice Market parameters
    struct MarketParams {
        address loanToken;
        address collateralToken;
        address oracle;
        address irm;
        uint256 lltv;
    }

    /// @notice Market state
    struct Market {
        uint128 totalSupplyAssets;
        uint128 totalSupplyShares;
        uint128 totalBorrowAssets;
        uint128 totalBorrowShares;
        uint128 lastUpdate;
        uint128 fee;
    }

    /// @notice Position state
    struct Position {
        uint256 supplyShares;
        uint128 borrowShares;
        uint128 collateral;
    }

    /// @notice Supply assets to a market
    function supply(
        MarketParams memory marketParams,
        uint256 assets,
        uint256 shares,
        address onBehalf,
        bytes memory data
    ) external returns (uint256 assetsSupplied, uint256 sharesSupplied);

    /// @notice Withdraw assets from a market
    function withdraw(
        MarketParams memory marketParams,
        uint256 assets,
        uint256 shares,
        address onBehalf,
        address receiver
    ) external returns (uint256 assetsWithdrawn, uint256 sharesWithdrawn);

    /// @notice Borrow assets from a market
    function borrow(
        MarketParams memory marketParams,
        uint256 assets,
        uint256 shares,
        address onBehalf,
        address receiver
    ) external returns (uint256 assetsBorrowed, uint256 sharesBorrowed);

    /// @notice Repay borrowed assets
    function repay(
        MarketParams memory marketParams,
        uint256 assets,
        uint256 shares,
        address onBehalf,
        bytes memory data
    ) external returns (uint256 assetsRepaid, uint256 sharesRepaid);

    /// @notice Supply collateral to a market
    function supplyCollateral(
        MarketParams memory marketParams,
        uint256 assets,
        address onBehalf,
        bytes memory data
    ) external;

    /// @notice Withdraw collateral from a market
    function withdrawCollateral(
        MarketParams memory marketParams,
        uint256 assets,
        address onBehalf,
        address receiver
    ) external;

    /// @notice Get market state by ID
    function market(bytes32 id) external view returns (Market memory);

    /// @notice Get position for an account in a market
    function position(bytes32 id, address account) external view returns (Position memory);

    /// @notice Get market ID from params
    function idFromMarketParams(MarketParams memory marketParams) external pure returns (bytes32);

    /// @notice Check if an account is authorized to manage another account's position
    function isAuthorized(address authorizer, address authorized) external view returns (bool);

    /// @notice Set authorization for an account
    function setAuthorization(address authorized, bool newIsAuthorized) external;
}
