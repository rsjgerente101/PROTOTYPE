import type { AlgorithmRun, ValidationResult, BaselineRunRequest, EnhancedRunRequest, } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export type FieldMapping = {
  depot_id?: string;
  depot_lat: string;
  depot_lon: string;
  customer_id: string;
  customer_lat: string;
  customer_lon: string;
  order_id?: string;
  eta_col?: string;
  rating_col?: string;
  area_col?: string;
};

export type DatasetRole =
  | 'primary_reconstruction'
  | 'comparative_template'
  | 'generic_uploaded_dataset';

export type ValidationResponse = ValidationResult & {
  datasetId: string;
  datasetRole: DatasetRole;
  sourceLabel: string;
  reconstructedBaselineReady: boolean;
  reconstructedBaselineName: string;
};

export type DatasetMetaResponse = {
  datasetId: string;
  filename: string;
  datasetRole: DatasetRole;
  sourceLabel: string;
  reconstructedBaselineName: string;
  records: number;
  depots: number;
  customers: number;
  orders: number;
  depot?: {
    id: string;
    lat: number;
    lon: number;
    name?: string;
  };
};

async function parseJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed with status ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export async function validateDataset(
  file: File,
  mapping: FieldMapping,
  datasetRole?: DatasetRole
): Promise<ValidationResponse> {
  const form = new FormData();
  form.append('file', file);
  form.append('mapping_json', JSON.stringify(mapping));

  if (datasetRole) {
    form.append('dataset_role', datasetRole);
  }

  const res = await fetch(`${API_BASE}/api/datasets/validate`, {
    method: 'POST',
    body: form,
  });

  return parseJson<ValidationResponse>(res);
}

export async function getDatasetMeta(datasetId: string): Promise<DatasetMetaResponse> {
  const res = await fetch(`${API_BASE}/api/datasets/${datasetId}/meta`);
  return parseJson<DatasetMetaResponse>(res);
}

export function getReconstructedDatasetDownloadUrl(datasetId: string): string {
  return `${API_BASE}/api/datasets/${datasetId}/reconstructed`;
}

export async function runBaseline(params: BaselineRunRequest): Promise<AlgorithmRun> {
  const res = await fetch(`${API_BASE}/api/runs/baseline`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_id: params.datasetId,
      num_representatives: params.numRepresentatives,
      avg_speed_kmph: params.avgSpeedKmph,
      service_minutes_per_stop: params.serviceMinutesPerStop,
      seed: params.seed,
      run_profile: params.runProfile,
    }),
  });

  return parseJson<AlgorithmRun>(res);
}

export async function runEnhanced(params: EnhancedRunRequest): Promise<AlgorithmRun> {
  const res = await fetch(`${API_BASE}/api/runs/enhanced`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset_id: params.datasetId,
      baseline_run_id: params.baselineRunId,
      fairness_weight: params.fairnessWeight,
      distance_weight: params.distanceWeight,
      time_weight: params.timeWeight,
      max_iterations: params.maxIterations,
      border_fraction: params.borderFraction,
      run_profile: params.runProfile,
    }),
  });

  return parseJson<AlgorithmRun>(res);
}

export async function checkBackendHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) return false;
    const json = await res.json();
    return json.status === 'ok';
  } catch {
    return false;
  }
}