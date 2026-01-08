/**
 * Input Validation Utilities
 *
 * Centralized validation for API endpoints and user inputs.
 */

import { isAddress } from 'viem';

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

export interface TransferInput {
  to?: unknown;
  amount?: unknown;
  proofHash?: unknown;
}

export interface WalletInput {
  userId?: unknown;
}

/**
 * Validate Ethereum address
 */
export function validateAddress(
  value: unknown,
  fieldName: string
): ValidationResult {
  const errors: string[] = [];

  if (value === undefined || value === null) {
    errors.push(`${fieldName} is required`);
    return { valid: false, errors };
  }

  if (typeof value !== 'string') {
    errors.push(`${fieldName} must be a string`);
    return { valid: false, errors };
  }

  if (!isAddress(value)) {
    errors.push(`${fieldName} is not a valid Ethereum address`);
    return { valid: false, errors };
  }

  return { valid: true, errors: [] };
}

/**
 * Validate positive number
 */
export function validatePositiveNumber(
  value: unknown,
  fieldName: string,
  options: { min?: number; max?: number } = {}
): ValidationResult {
  const errors: string[] = [];

  if (value === undefined || value === null) {
    errors.push(`${fieldName} is required`);
    return { valid: false, errors };
  }

  const num = typeof value === 'string' ? parseFloat(value) : value;

  if (typeof num !== 'number' || isNaN(num)) {
    errors.push(`${fieldName} must be a valid number`);
    return { valid: false, errors };
  }

  if (num <= 0) {
    errors.push(`${fieldName} must be positive`);
    return { valid: false, errors };
  }

  if (options.min !== undefined && num < options.min) {
    errors.push(`${fieldName} must be at least ${options.min}`);
    return { valid: false, errors };
  }

  if (options.max !== undefined && num > options.max) {
    errors.push(`${fieldName} must be at most ${options.max}`);
    return { valid: false, errors };
  }

  return { valid: true, errors: [] };
}

/**
 * Validate proof hash (bytes32)
 */
export function validateProofHash(
  value: unknown,
  fieldName: string,
  required = false
): ValidationResult {
  const errors: string[] = [];

  if (value === undefined || value === null || value === '') {
    if (required) {
      errors.push(`${fieldName} is required`);
      return { valid: false, errors };
    }
    return { valid: true, errors: [] };
  }

  if (typeof value !== 'string') {
    errors.push(`${fieldName} must be a string`);
    return { valid: false, errors };
  }

  // Check format: 0x followed by 64 hex characters
  const hashRegex = /^0x[a-fA-F0-9]{64}$/;
  if (!hashRegex.test(value)) {
    errors.push(
      `${fieldName} must be a valid bytes32 hash (0x + 64 hex characters)`
    );
    return { valid: false, errors };
  }

  return { valid: true, errors: [] };
}

/**
 * Validate string with optional length constraints
 */
export function validateString(
  value: unknown,
  fieldName: string,
  options: { minLength?: number; maxLength?: number; required?: boolean } = {}
): ValidationResult {
  const errors: string[] = [];
  const { minLength, maxLength, required = true } = options;

  if (value === undefined || value === null || value === '') {
    if (required) {
      errors.push(`${fieldName} is required`);
      return { valid: false, errors };
    }
    return { valid: true, errors: [] };
  }

  if (typeof value !== 'string') {
    errors.push(`${fieldName} must be a string`);
    return { valid: false, errors };
  }

  if (minLength !== undefined && value.length < minLength) {
    errors.push(`${fieldName} must be at least ${minLength} characters`);
    return { valid: false, errors };
  }

  if (maxLength !== undefined && value.length > maxLength) {
    errors.push(`${fieldName} must be at most ${maxLength} characters`);
    return { valid: false, errors };
  }

  return { valid: true, errors: [] };
}

/**
 * Validate transfer request input
 */
export function validateTransferInput(input: TransferInput): ValidationResult {
  const errors: string[] = [];

  const toResult = validateAddress(input.to, 'to');
  if (!toResult.valid) {
    errors.push(...toResult.errors);
  }

  const amountResult = validatePositiveNumber(input.amount, 'amount', {
    min: 0.000001,
    max: 1000000,
  });
  if (!amountResult.valid) {
    errors.push(...amountResult.errors);
  }

  // proofHash is optional
  if (input.proofHash !== undefined && input.proofHash !== null) {
    const proofResult = validateProofHash(input.proofHash, 'proofHash');
    if (!proofResult.valid) {
      errors.push(...proofResult.errors);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Validate wallet creation input
 */
export function validateWalletInput(input: WalletInput): ValidationResult {
  const errors: string[] = [];

  if (input.userId !== undefined) {
    const userIdResult = validateString(input.userId, 'userId', {
      minLength: 1,
      maxLength: 100,
      required: false,
    });
    if (!userIdResult.valid) {
      errors.push(...userIdResult.errors);
    }
  }

  return {
    valid: errors.length === 0,
    errors,
  };
}

/**
 * Create a standardized API error response
 */
export interface ApiErrorResponse {
  success: false;
  error: string;
  code: string;
  details?: string[];
}

export function createValidationErrorResponse(
  errors: string[]
): ApiErrorResponse {
  return {
    success: false,
    error: errors.length === 1 ? errors[0] : 'Validation failed',
    code: 'VALIDATION_ERROR',
    details: errors.length > 1 ? errors : undefined,
  };
}

export function createErrorResponse(
  error: string,
  code: string
): ApiErrorResponse {
  return {
    success: false,
    error,
    code,
  };
}
