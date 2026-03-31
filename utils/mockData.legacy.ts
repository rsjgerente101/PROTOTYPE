import { Dataset, Route, KPIMetrics, Representative, AlgorithmRun, RouteStop } from '../src/types';
export const MOCK_DATASET: Dataset = {
  depots: [{
    id: 'D001',
    lat: 14.5995,
    lon: 120.9842,
    name: 'Manila Central Depot'
  }],
  customers: [{
    id: 'C001',
    lat: 14.6042,
    lon: 120.9822,
    name: 'Customer A',
    orderId: 'ORD001'
  }, {
    id: 'C002',
    lat: 14.5952,
    lon: 120.9912,
    name: 'Customer B',
    orderId: 'ORD002'
  }, {
    id: 'C003',
    lat: 14.6102,
    lon: 120.9752,
    name: 'Customer C',
    orderId: 'ORD003'
  }, {
    id: 'C004',
    lat: 14.5892,
    lon: 120.9982,
    name: 'Customer D',
    orderId: 'ORD004'
  }, {
    id: 'C005',
    lat: 14.6152,
    lon: 120.9882,
    name: 'Customer E',
    orderId: 'ORD005'
  }, {
    id: 'C006',
    lat: 14.5852,
    lon: 120.9752,
    name: 'Customer F',
    orderId: 'ORD006'
  }, {
    id: 'C007',
    lat: 14.6202,
    lon: 120.9812,
    name: 'Customer G',
    orderId: 'ORD007'
  }, {
    id: 'C008',
    lat: 14.5802,
    lon: 120.9912,
    name: 'Customer H',
    orderId: 'ORD008'
  }, {
    id: 'C009',
    lat: 14.6252,
    lon: 120.9742,
    name: 'Customer I',
    orderId: 'ORD009'
  }, {
    id: 'C010',
    lat: 14.5752,
    lon: 120.9852,
    name: 'Customer J',
    orderId: 'ORD010'
  }, {
    id: 'C011',
    lat: 14.6302,
    lon: 120.9892,
    name: 'Customer K',
    orderId: 'ORD011'
  }, {
    id: 'C012',
    lat: 14.5702,
    lon: 120.9782,
    name: 'Customer L',
    orderId: 'ORD012'
  }],
  totalOrders: 12
};
const REPRESENTATIVE_COLORS = ['#3B82F6', '#F97316', '#10B981', '#8B5CF6'];
export const generateBaselineRun = (): AlgorithmRun => {
  const representatives: Representative[] = [{
    id: 'REP001',
    name: 'John Martinez',
    workload: 3.2,
    opportunityScore: 85,
    priorityScore: 7.5,
    color: REPRESENTATIVE_COLORS[0],
    assignedCustomers: ['C001', 'C002', 'C003']
  }, {
    id: 'REP002',
    name: 'Maria Santos',
    workload: 4.1,
    opportunityScore: 72,
    priorityScore: 6.8,
    color: REPRESENTATIVE_COLORS[1],
    assignedCustomers: ['C004', 'C005', 'C006', 'C007']
  }, {
    id: 'REP003',
    name: 'Carlos Reyes',
    workload: 2.8,
    opportunityScore: 91,
    priorityScore: 8.2,
    color: REPRESENTATIVE_COLORS[2],
    assignedCustomers: ['C008', 'C009', 'C010']
  }, {
    id: 'REP004',
    name: 'Ana Cruz',
    workload: 2.5,
    opportunityScore: 78,
    priorityScore: 7.1,
    color: REPRESENTATIVE_COLORS[3],
    assignedCustomers: ['C011', 'C012']
  }];
  const routes: Route[] = representatives.map((rep, idx) => {
    const customers = rep.assignedCustomers.map(custId => MOCK_DATASET.customers.find(c => c.id === custId)!);
    const stops: RouteStop[] = [{
      stopNumber: 0,
      nodeId: MOCK_DATASET.depots[0].id,
      nodeName: MOCK_DATASET.depots[0].name,
      legDistance: 0,
      cumulativeDistance: 0,
      eta: '08:00',
      lat: MOCK_DATASET.depots[0].lat,
      lon: MOCK_DATASET.depots[0].lon
    }, ...customers.map((cust, i) => ({
      stopNumber: i + 1,
      nodeId: cust.id,
      nodeName: cust.name || cust.id,
      legDistance: 2.3 + Math.random() * 3,
      cumulativeDistance: (i + 1) * 3.5 + Math.random() * 5,
      eta: `${8 + Math.floor((i + 1) * 0.5)}:${i * 15 % 60}`,
      lat: cust.lat,
      lon: cust.lon
    }))];
    return {
      id: `ROUTE_${rep.id}`,
      representativeId: rep.id,
      representativeName: rep.name,
      stops,
      totalDistance: stops[stops.length - 1].cumulativeDistance,
      totalTime: customers.length * 0.75 + 1.5,
      color: rep.color
    };
  });
  const kpis: KPIMetrics = {
    totalDistance: routes.reduce((sum, r) => sum + r.totalDistance, 0),
    totalTime: routes.reduce((sum, r) => sum + r.totalTime, 0),
    computeTime: 342,
    numberOfStops: routes.reduce((sum, r) => sum + r.stops.length - 1, 0),
    delayScore: 24.5,
    ratingPenalty: 8.2,
    coverageRatio: 0.83,
    workloadBalanceIndex: 0.72,
    jainsFairnessIndex: 0.68
  };
  return {
    type: 'baseline',
    routes,
    kpis,
    representatives,
    timestamp: new Date().toISOString()
  };
};
export const generateEnhancedRun = (baselineRun: AlgorithmRun): AlgorithmRun => {
  const representatives: Representative[] = [{
    id: 'REP001',
    name: 'John Martinez',
    workload: 3.5,
    opportunityScore: 88,
    priorityScore: 8.1,
    color: REPRESENTATIVE_COLORS[0],
    assignedCustomers: ['C001', 'C002', 'C003', 'C012']
  }, {
    id: 'REP002',
    name: 'Maria Santos',
    workload: 3.4,
    opportunityScore: 85,
    priorityScore: 7.9,
    color: REPRESENTATIVE_COLORS[1],
    assignedCustomers: ['C004', 'C005', 'C006']
  }, {
    id: 'REP003',
    name: 'Carlos Reyes',
    workload: 3.2,
    opportunityScore: 92,
    priorityScore: 8.5,
    color: REPRESENTATIVE_COLORS[2],
    assignedCustomers: ['C008', 'C009', 'C010']
  }, {
    id: 'REP004',
    name: 'Ana Cruz',
    workload: 3.1,
    opportunityScore: 84,
    priorityScore: 8.0,
    color: REPRESENTATIVE_COLORS[3],
    assignedCustomers: ['C007', 'C011']
  }];
  const routes: Route[] = representatives.map(rep => {
    const customers = rep.assignedCustomers.map(custId => MOCK_DATASET.customers.find(c => c.id === custId)!);
    const stops: RouteStop[] = [{
      stopNumber: 0,
      nodeId: MOCK_DATASET.depots[0].id,
      nodeName: MOCK_DATASET.depots[0].name,
      legDistance: 0,
      cumulativeDistance: 0,
      eta: '08:00',
      lat: MOCK_DATASET.depots[0].lat,
      lon: MOCK_DATASET.depots[0].lon
    }, ...customers.map((cust, i) => ({
      stopNumber: i + 1,
      nodeId: cust.id,
      nodeName: cust.name || cust.id,
      legDistance: 2.1 + Math.random() * 2.5,
      cumulativeDistance: (i + 1) * 3.2 + Math.random() * 4,
      eta: `${8 + Math.floor((i + 1) * 0.5)}:${i * 15 % 60}`,
      lat: cust.lat,
      lon: cust.lon
    }))];
    return {
      id: `ROUTE_${rep.id}`,
      representativeId: rep.id,
      representativeName: rep.name,
      stops,
      totalDistance: stops[stops.length - 1].cumulativeDistance,
      totalTime: customers.length * 0.7 + 1.2,
      color: rep.color
    };
  });
  const kpis: KPIMetrics = {
    totalDistance: routes.reduce((sum, r) => sum + r.totalDistance, 0),
    totalTime: routes.reduce((sum, r) => sum + r.totalTime, 0),
    computeTime: 418,
    numberOfStops: routes.reduce((sum, r) => sum + r.stops.length - 1, 0),
    delayScore: 18.3,
    ratingPenalty: 5.1,
    coverageRatio: 0.91,
    workloadBalanceIndex: 0.89,
    jainsFairnessIndex: 0.87
  };
  return {
    type: 'enhanced',
    routes,
    kpis,
    representatives,
    timestamp: new Date().toISOString()
  };
};