/**
 * OpenMind Robot Demo - Mock Data
 *
 * Sample robots, services, and policies for the demo
 */

import { generateSecureBytes32 } from '@/lib/crypto';
import type {
  RobotSpendingPolicy,
  RobotWalletState,
  ServicePayment,
  RobotStatus,
  RobotType,
  ServiceCategory,
} from './types';

/**
 * Sample robot configurations
 */
export const SAMPLE_ROBOTS: RobotStatus[] = [
  {
    id: 'robot-001',
    name: 'DeliveryBot-7',
    type: 'delivery',
    status: 'idle',
    batteryLevel: 73,
    location: 'San Francisco, CA',
    currentTask: 'Package delivery route',
  },
  {
    id: 'robot-002',
    name: 'InspectorDrone-3',
    type: 'inspection',
    status: 'idle',
    batteryLevel: 45,
    location: 'Palo Alto, CA',
    currentTask: 'Infrastructure inspection',
  },
  {
    id: 'robot-003',
    name: 'HomeAssist-12',
    type: 'home_assistant',
    status: 'idle',
    batteryLevel: 89,
    location: 'Mountain View, CA',
    currentTask: 'Household management',
  },
];

/**
 * Default spending policy for robots
 */
export const DEFAULT_ROBOT_POLICY: RobotSpendingPolicy = {
  dailyLimitUsdc: 10.0,
  maxSingleTxUsdc: 2.0,
  allowedCategories: ['charging', 'navigation', 'compute', 'data'],
  minServiceReliability: 0.85,
  requireProofForAll: true,
  emergencyThresholdUsdc: 5.0,
};

/**
 * Conservative policy for home robots
 */
export const CONSERVATIVE_POLICY: RobotSpendingPolicy = {
  dailyLimitUsdc: 5.0,
  maxSingleTxUsdc: 1.0,
  allowedCategories: ['charging', 'data'],
  minServiceReliability: 0.95,
  requireProofForAll: true,
};

/**
 * Enterprise policy for commercial robots
 */
export const ENTERPRISE_POLICY: RobotSpendingPolicy = {
  dailyLimitUsdc: 50.0,
  maxSingleTxUsdc: 10.0,
  allowedCategories: ['charging', 'navigation', 'compute', 'data', 'transport', 'maintenance'],
  minServiceReliability: 0.80,
  requireProofForAll: true,
  emergencyThresholdUsdc: 25.0,
};

/**
 * Generate a mock wallet state
 */
export function createMockWalletState(spentToday: number = 0): RobotWalletState {
  return {
    address: generateSecureBytes32() as `0x${string}`,
    balanceUsdc: 100.0,
    spentTodayUsdc: spentToday,
    txCountToday: Math.floor(spentToday / 0.5),
    lastTxTimestamp: spentToday > 0 ? Date.now() - 3600000 : undefined,
  };
}

/**
 * Sample services that robots can pay for
 */
export const SAMPLE_SERVICES: ServicePayment[] = [
  {
    serviceId: 'charging-station-sf-001',
    serviceName: 'ChargePoint Station #4521',
    category: 'charging',
    priceUsdc: 0.10,
    reliabilityScore: 0.98,
    description: 'Fast charging session (15 min, 50kW)',
    providerAddress: '0x8ba1f109551bD432803012645Ac136ddd64DBA72',
  },
  {
    serviceId: 'nav-api-mapbox',
    serviceName: 'Mapbox Navigation API',
    category: 'navigation',
    priceUsdc: 0.004,
    reliabilityScore: 0.99,
    description: 'Real-time route calculation with traffic',
    providerAddress: '0x742d35Cc6634C0532925a3b844Bc9e7595f4bF8E',
  },
  {
    serviceId: 'compute-aws-lambda',
    serviceName: 'AWS Lambda Compute',
    category: 'compute',
    priceUsdc: 0.02,
    reliabilityScore: 0.9999,
    description: 'Vision processing inference (100ms)',
    providerAddress: '0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed',
  },
  {
    serviceId: 'weather-data-api',
    serviceName: 'OpenWeather API',
    category: 'data',
    priceUsdc: 0.001,
    reliabilityScore: 0.97,
    description: 'Current weather conditions',
    providerAddress: '0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359',
  },
  {
    serviceId: 'waymo-ride-request',
    serviceName: 'Waymo Transport',
    category: 'transport',
    priceUsdc: 15.00,
    reliabilityScore: 0.96,
    description: 'Autonomous vehicle transport (5 miles)',
    providerAddress: '0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe',
  },
  {
    serviceId: 'maintenance-diagnostic',
    serviceName: 'RoboMaint Diagnostics',
    category: 'maintenance',
    priceUsdc: 5.00,
    reliabilityScore: 0.92,
    description: 'Full system diagnostic scan',
    providerAddress: '0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8',
  },
];

/**
 * Demo scenario: Delivery robot needs to charge
 */
export const CHARGING_SCENARIO = {
  robot: SAMPLE_ROBOTS[0],
  service: SAMPLE_SERVICES[0], // ChargePoint charging
  policy: DEFAULT_ROBOT_POLICY,
  walletState: createMockWalletState(3.50), // Already spent $3.50 today
  expectedApproval: true,
  reason: 'Within daily limit, approved category, high reliability',
};

/**
 * Demo scenario: Robot tries expensive transport (should be rejected)
 */
export const TRANSPORT_SCENARIO = {
  robot: SAMPLE_ROBOTS[0],
  service: SAMPLE_SERVICES[4], // Waymo $15
  policy: DEFAULT_ROBOT_POLICY,
  walletState: createMockWalletState(2.00),
  expectedApproval: false,
  reason: 'Exceeds max single transaction limit ($2)',
};

/**
 * Demo scenario: Robot near daily limit
 */
export const NEAR_LIMIT_SCENARIO = {
  robot: SAMPLE_ROBOTS[1],
  service: SAMPLE_SERVICES[2], // AWS compute $0.02
  policy: DEFAULT_ROBOT_POLICY,
  walletState: createMockWalletState(9.50), // Almost at $10 limit
  expectedApproval: true,
  reason: 'Small transaction, within remaining daily budget',
};

/**
 * Get category display name
 */
export function getCategoryDisplayName(category: ServiceCategory): string {
  const names: Record<ServiceCategory, string> = {
    charging: 'Charging',
    navigation: 'Navigation',
    compute: 'Compute',
    data: 'Data',
    transport: 'Transport',
    maintenance: 'Maintenance',
    storage: 'Storage',
    communication: 'Communication',
    other: 'Other',
  };
  return names[category];
}

/**
 * Get robot type display name
 */
export function getRobotTypeDisplayName(type: RobotType): string {
  const names: Record<RobotType, string> = {
    humanoid: 'Humanoid',
    quadruped: 'Quadruped',
    delivery: 'Delivery Bot',
    inspection: 'Inspection Drone',
    home_assistant: 'Home Assistant',
    digital_agent: 'Digital Agent',
  };
  return names[type];
}

/**
 * Get robot type icon name (for lucide-react)
 */
export function getRobotTypeIcon(type: RobotType): string {
  const icons: Record<RobotType, string> = {
    humanoid: 'Bot',
    quadruped: 'Dog',
    delivery: 'Package',
    inspection: 'Scan',
    home_assistant: 'Home',
    digital_agent: 'Cpu',
  };
  return icons[type];
}
