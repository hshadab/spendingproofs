// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "../interfaces/IMorphoBlue.sol";

/**
 * @title MockMorphoBlue
 * @notice Mock Morpho Blue for testing
 */
contract MockMorphoBlue is IMorphoBlue {
    mapping(bytes32 => Market) public markets;
    mapping(bytes32 => mapping(address => Position)) public positions;
    mapping(address => mapping(address => bool)) public authorizations;

    uint256 public constant SHARES_MULTIPLIER = 1e18;

    function supply(
        MarketParams memory marketParams,
        uint256 assets,
        uint256,
        address onBehalf,
        bytes memory
    ) external override returns (uint256 assetsSupplied, uint256 sharesSupplied) {
        bytes32 id = idFromMarketParams(marketParams);
        sharesSupplied = assets * SHARES_MULTIPLIER / 1e18;

        markets[id].totalSupplyAssets += uint128(assets);
        markets[id].totalSupplyShares += uint128(sharesSupplied);
        positions[id][onBehalf].supplyShares += sharesSupplied;

        return (assets, sharesSupplied);
    }

    function withdraw(
        MarketParams memory marketParams,
        uint256 assets,
        uint256,
        address onBehalf,
        address
    ) external override returns (uint256 assetsWithdrawn, uint256 sharesWithdrawn) {
        bytes32 id = idFromMarketParams(marketParams);
        sharesWithdrawn = assets * SHARES_MULTIPLIER / 1e18;

        markets[id].totalSupplyAssets -= uint128(assets);
        markets[id].totalSupplyShares -= uint128(sharesWithdrawn);
        positions[id][onBehalf].supplyShares -= sharesWithdrawn;

        return (assets, sharesWithdrawn);
    }

    function borrow(
        MarketParams memory marketParams,
        uint256 assets,
        uint256,
        address onBehalf,
        address
    ) external override returns (uint256 assetsBorrowed, uint256 sharesBorrowed) {
        bytes32 id = idFromMarketParams(marketParams);
        sharesBorrowed = assets * SHARES_MULTIPLIER / 1e18;

        markets[id].totalBorrowAssets += uint128(assets);
        markets[id].totalBorrowShares += uint128(sharesBorrowed);
        positions[id][onBehalf].borrowShares += uint128(sharesBorrowed);

        return (assets, sharesBorrowed);
    }

    function repay(
        MarketParams memory marketParams,
        uint256 assets,
        uint256,
        address onBehalf,
        bytes memory
    ) external override returns (uint256 assetsRepaid, uint256 sharesRepaid) {
        bytes32 id = idFromMarketParams(marketParams);
        sharesRepaid = assets * SHARES_MULTIPLIER / 1e18;

        markets[id].totalBorrowAssets -= uint128(assets);
        markets[id].totalBorrowShares -= uint128(sharesRepaid);
        positions[id][onBehalf].borrowShares -= uint128(sharesRepaid);

        return (assets, sharesRepaid);
    }

    function supplyCollateral(
        MarketParams memory marketParams,
        uint256 assets,
        address onBehalf,
        bytes memory
    ) external override {
        bytes32 id = idFromMarketParams(marketParams);
        positions[id][onBehalf].collateral += uint128(assets);
    }

    function withdrawCollateral(
        MarketParams memory marketParams,
        uint256 assets,
        address onBehalf,
        address
    ) external override {
        bytes32 id = idFromMarketParams(marketParams);
        positions[id][onBehalf].collateral -= uint128(assets);
    }

    function market(bytes32 id) external view override returns (Market memory) {
        return markets[id];
    }

    function position(bytes32 id, address account) external view override returns (Position memory) {
        return positions[id][account];
    }

    function idFromMarketParams(MarketParams memory marketParams) public pure override returns (bytes32) {
        return keccak256(abi.encode(marketParams));
    }

    function isAuthorized(address authorizer, address authorized) external view override returns (bool) {
        return authorizations[authorizer][authorized];
    }

    function setAuthorization(address authorized, bool newIsAuthorized) external override {
        authorizations[msg.sender][authorized] = newIsAuthorized;
    }
}
