/**
 * Policy DSL for defining spending rules
 * Supports nested AND/OR conditions with various operators
 */

export type Operator = 'eq' | 'neq' | 'gt' | 'gte' | 'lt' | 'lte' | 'in' | 'between';

export type FieldType =
  | 'price'
  | 'merchantTrust'
  | 'categoryMatch'
  | 'urgency'
  | 'volatility'
  | 'stockLevel'
  | 'promotionalDiscount'
  | 'historicalAccuracy';

export interface Condition {
  id: string;
  field: FieldType;
  operator: Operator;
  value: number | number[];
}

export interface ConditionGroup {
  id: string;
  type: 'AND' | 'OR';
  conditions: (Condition | ConditionGroup)[];
}

export interface PolicyRule {
  id: string;
  name: string;
  description: string;
  root: ConditionGroup;
  action: 'approve' | 'reject';
  priority: number;
}

export interface Policy {
  id: string;
  name: string;
  version: number;
  rules: PolicyRule[];
  defaultAction: 'approve' | 'reject';
  createdAt: number;
  updatedAt: number;
}

// Field metadata
export const FIELD_METADATA: Record<FieldType, {
  label: string;
  description: string;
  unit: string;
  min: number;
  max: number;
  step: number;
}> = {
  price: {
    label: 'Price',
    description: 'Transaction amount in USD',
    unit: '$',
    min: 0,
    max: 10000,
    step: 1,
  },
  merchantTrust: {
    label: 'Merchant Trust',
    description: 'Trust score of the merchant (0-1)',
    unit: '%',
    min: 0,
    max: 1,
    step: 0.01,
  },
  categoryMatch: {
    label: 'Category Match',
    description: 'How well the item matches allowed categories',
    unit: '%',
    min: 0,
    max: 1,
    step: 0.01,
  },
  urgency: {
    label: 'Urgency',
    description: 'How urgent is the purchase',
    unit: '%',
    min: 0,
    max: 1,
    step: 0.01,
  },
  volatility: {
    label: 'Price Volatility',
    description: 'Recent price volatility of the item',
    unit: '%',
    min: 0,
    max: 1,
    step: 0.01,
  },
  stockLevel: {
    label: 'Stock Level',
    description: 'Current stock availability',
    unit: '%',
    min: 0,
    max: 1,
    step: 0.01,
  },
  promotionalDiscount: {
    label: 'Discount',
    description: 'Applied promotional discount',
    unit: '%',
    min: 0,
    max: 1,
    step: 0.01,
  },
  historicalAccuracy: {
    label: 'Historical Accuracy',
    description: 'Accuracy of past predictions',
    unit: '%',
    min: 0,
    max: 1,
    step: 0.01,
  },
};

export const OPERATOR_LABELS: Record<Operator, string> = {
  eq: 'equals',
  neq: 'not equals',
  gt: 'greater than',
  gte: 'greater than or equal',
  lt: 'less than',
  lte: 'less than or equal',
  in: 'is one of',
  between: 'is between',
};

// Evaluate a single condition
export function evaluateCondition(condition: Condition, values: Record<FieldType, number>): boolean {
  const fieldValue = values[condition.field];
  const targetValue = condition.value;

  switch (condition.operator) {
    case 'eq':
      return fieldValue === targetValue;
    case 'neq':
      return fieldValue !== targetValue;
    case 'gt':
      return fieldValue > (targetValue as number);
    case 'gte':
      return fieldValue >= (targetValue as number);
    case 'lt':
      return fieldValue < (targetValue as number);
    case 'lte':
      return fieldValue <= (targetValue as number);
    case 'in':
      return (targetValue as number[]).includes(fieldValue);
    case 'between':
      const [min, max] = targetValue as number[];
      return fieldValue >= min && fieldValue <= max;
    default:
      return false;
  }
}

// Check if item is a condition or group
export function isConditionGroup(item: Condition | ConditionGroup): item is ConditionGroup {
  return 'type' in item && 'conditions' in item;
}

// Evaluate a condition group
export function evaluateConditionGroup(
  group: ConditionGroup,
  values: Record<FieldType, number>
): boolean {
  const results = group.conditions.map((item) => {
    if (isConditionGroup(item)) {
      return evaluateConditionGroup(item, values);
    }
    return evaluateCondition(item, values);
  });

  if (group.type === 'AND') {
    return results.every(Boolean);
  }
  return results.some(Boolean);
}

// Evaluate a policy rule
export function evaluateRule(rule: PolicyRule, values: Record<FieldType, number>): boolean {
  return evaluateConditionGroup(rule.root, values);
}

// Evaluate entire policy and return action
export function evaluatePolicy(
  policy: Policy,
  values: Record<FieldType, number>
): { action: 'approve' | 'reject'; matchedRule: PolicyRule | null } {
  // Sort rules by priority (higher priority first)
  const sortedRules = [...policy.rules].sort((a, b) => b.priority - a.priority);

  for (const rule of sortedRules) {
    if (evaluateRule(rule, values)) {
      return { action: rule.action, matchedRule: rule };
    }
  }

  return { action: policy.defaultAction, matchedRule: null };
}

// Generate unique ID
export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

// Create a new empty condition
export function createCondition(field: FieldType = 'price'): Condition {
  return {
    id: generateId(),
    field,
    operator: 'lte',
    value: FIELD_METADATA[field].max / 2,
  };
}

// Create a new condition group
export function createConditionGroup(type: 'AND' | 'OR' = 'AND'): ConditionGroup {
  return {
    id: generateId(),
    type,
    conditions: [createCondition()],
  };
}

// Create a new rule
export function createRule(): PolicyRule {
  return {
    id: generateId(),
    name: 'New Rule',
    description: '',
    root: createConditionGroup('AND'),
    action: 'approve',
    priority: 1,
  };
}

// Create a new policy
export function createPolicy(name: string = 'New Policy'): Policy {
  return {
    id: generateId(),
    name,
    version: 1,
    rules: [createRule()],
    defaultAction: 'reject',
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

// Serialize policy to JSON
export function serializePolicy(policy: Policy): string {
  return JSON.stringify(policy, null, 2);
}

// Deserialize policy from JSON
export function deserializePolicy(json: string): Policy {
  return JSON.parse(json) as Policy;
}

// Example policies
export const EXAMPLE_POLICIES: Policy[] = [
  {
    id: 'conservative',
    name: 'Conservative Spending',
    version: 1,
    rules: [
      {
        id: 'low-value-trusted',
        name: 'Low Value Trusted Purchases',
        description: 'Approve low-value purchases from trusted merchants',
        root: {
          id: 'root-1',
          type: 'AND',
          conditions: [
            { id: 'c1', field: 'price', operator: 'lte', value: 100 },
            { id: 'c2', field: 'merchantTrust', operator: 'gte', value: 0.7 },
          ],
        },
        action: 'approve',
        priority: 2,
      },
      {
        id: 'high-value-very-trusted',
        name: 'High Value Very Trusted',
        description: 'Approve higher values only from very trusted merchants',
        root: {
          id: 'root-2',
          type: 'AND',
          conditions: [
            { id: 'c3', field: 'price', operator: 'between', value: [100, 500] },
            { id: 'c4', field: 'merchantTrust', operator: 'gte', value: 0.9 },
            { id: 'c5', field: 'categoryMatch', operator: 'gte', value: 0.8 },
          ],
        },
        action: 'approve',
        priority: 1,
      },
    ],
    defaultAction: 'reject',
    createdAt: Date.now(),
    updatedAt: Date.now(),
  },
  {
    id: 'balanced',
    name: 'Balanced Spending',
    version: 1,
    rules: [
      {
        id: 'standard-approval',
        name: 'Standard Approval',
        description: 'Approve if meets basic criteria',
        root: {
          id: 'root-3',
          type: 'AND',
          conditions: [
            { id: 'c6', field: 'price', operator: 'lte', value: 500 },
            { id: 'c7', field: 'merchantTrust', operator: 'gte', value: 0.5 },
            {
              id: 'nested-1',
              type: 'OR',
              conditions: [
                { id: 'c8', field: 'categoryMatch', operator: 'gte', value: 0.7 },
                { id: 'c9', field: 'urgency', operator: 'gte', value: 0.8 },
              ],
            },
          ],
        },
        action: 'approve',
        priority: 1,
      },
    ],
    defaultAction: 'reject',
    createdAt: Date.now(),
    updatedAt: Date.now(),
  },
];
