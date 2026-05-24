import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { Select } from '../components/Select';
import type { Dataset } from '../types';
import {
  checkBackendHealth,
  getReconstructedDatasetDownloadUrl,
  validateDataset,
  type DatasetRole,
  type FieldMapping,
  type ValidationResponse,
} from '../services/deliveryApi';

const STORAGE_FILE_KEY = 'uploadedDatasetFileMeta';
const STORAGE_VALIDATION_KEY = 'uploadedDatasetValidation';

const FIELD_OPTIONS = [
  { label: 'Depot ID', value: 'depot_id' },
  { label: 'Depot Latitude', value: 'depot_lat' },
  { label: 'Depot Longitude', value: 'depot_lon' },
  { label: 'Customer Latitude', value: 'customer_lat' },
  { label: 'Customer Longitude', value: 'customer_lon' },
  { label: 'Order ID', value: 'order_id' },
  { label: 'Order Date', value: 'order_date_col' },
  { label: 'ETA Column', value: 'eta_col' },
  { label: 'Rating Column', value: 'rating_col' },
  { label: 'Area/Cluster Column', value: 'area_col' },
];

const REQUIRED_RECONSTRUCTED_HEADERS = [
  'depot_lat',
  'depot_lon',
  'customer_id',
  'customer_lat',
  'customer_lon',
];

function cleanHeader(value: string): string {
  return value
    .replace(/^\uFEFF/, '')
    .trim()
    .toLowerCase()
    .replace(/^"|"$/g, '')
    .replace(/[_\s]+/g, '_');
}

function inferFrontendDatasetRole(filename: string): DatasetRole {
  const name = filename.toLowerCase();
  if (name.includes('amazon')) return 'primary_reconstruction';
  if (name.includes('zomato')) return 'comparative_template';
  return 'generic_uploaded_dataset';
}

function roleBadgeText(role: DatasetRole): string {
  if (role === 'primary_reconstruction') return 'Primary Baseline Reconstruction Source';
  if (role === 'comparative_template') return 'Comparative Template Dataset';
  return 'Generic Uploaded Dataset';
}

function isReconstructedSchema(headers: string[]): boolean {
  const normalized = headers.map(cleanHeader);
  return REQUIRED_RECONSTRUCTED_HEADERS.every((required) => normalized.includes(required));
}

function buildAutoMapping(headers: string[]): FieldMapping {
  const byNormalized = new Map(headers.map((h) => [cleanHeader(h), h]));

  return {
    depot_id: byNormalized.get('depot_id') || undefined,
    depot_lat: byNormalized.get('depot_lat') || '',
    depot_lon: byNormalized.get('depot_lon') || '',
    customer_id: byNormalized.get('customer_id') || '',
    agent_id: byNormalized.get('agent_id') || '',
    customer_lat: byNormalized.get('customer_lat') || '',
    customer_lon: byNormalized.get('customer_lon') || '',
    order_id: byNormalized.get('order_id') || undefined,
    order_date_col:
      byNormalized.get('order_date') || byNormalized.get('order_date_col') || undefined,
    eta_col: byNormalized.get('observed_eta_min') || byNormalized.get('eta') || undefined,
    rating_col: byNormalized.get('rating') || undefined,
    area_col: byNormalized.get('area') || undefined,
  };
}

const DatasetUpload: React.FC = () => {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [csvHeaders, setCsvHeaders] = useState<string[]>([]);
  const [mapping, setMapping] = useState<FieldMapping>({
    depot_lat: '',
    depot_lon: '',
    customer_id: '',
    customer_lat: '',
    customer_lon: '',
    agent_id: '',
  });
  const [autoDetectedReconstructed, setAutoDetectedReconstructed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [backendReady, setBackendReady] = useState<boolean | null>(null);
  const [message, setMessage] = useState('');
  const [validation, setValidation] = useState<ValidationResponse | null>(null);

  useEffect(() => {
    checkBackendHealth().then(setBackendReady);
  }, []);

  const inferredRole = useMemo<DatasetRole | null>(() => {
    if (!file) return null;
    return inferFrontendDatasetRole(file.name);
  }, [file]);

  const parseCsvHeaders = async (uploadedFile: File): Promise<string[]> => {
    const text = await uploadedFile.text();
    const firstLine = text.split(/\r?\n/)[0] ?? '';

    let delimiter = ',';
    if (firstLine.includes('\t')) {
      delimiter = '\t';
    } else if (firstLine.includes(';')) {
      delimiter = ';';
    }

    const headers = firstLine
      .split(delimiter)
      .map((h) => h.trim())
      .filter(Boolean);

    setCsvHeaders(headers);
    return headers;
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = e.target.files?.[0] ?? null;
    setFile(uploadedFile);
    setValidation(null);
    setMessage('');
    setAutoDetectedReconstructed(false);

    if (!uploadedFile) {
      setCsvHeaders([]);
      setMapping({
        depot_lat: '',
        depot_lon: '',
        customer_id: '',
        customer_lat: '',
        customer_lon: '',
        agent_id: '',
      });
      return;
    }

    try {
      const headers = await parseCsvHeaders(uploadedFile);

      const inferred = inferFrontendDatasetRole(uploadedFile.name);

      if (isReconstructedSchema(headers)) {
        setAutoDetectedReconstructed(true);
        setMapping(buildAutoMapping(headers));
        setMessage(
          `Loaded file: ${uploadedFile.name}. Detected role: ${roleBadgeText(
            inferred
          )}. Standardized reconstructed baseline dataset detected, so manual field mapping is not required.`
        );
      } else {
        setMapping({
          depot_lat: '',
          depot_lon: '',
          customer_id: '',
          customer_lat: '',
          customer_lon: '',
          agent_id: '',
        });
        setMessage(
          `Loaded file: ${uploadedFile.name}. Detected role: ${roleBadgeText(
            inferred
          )}. Map the required fields below.`
        );
      }
    } catch {
      setCsvHeaders([]);
      setAutoDetectedReconstructed(false);
      setMessage('Failed to read CSV headers.');
    }
  };

  const onValidate = async () => {
    if (!file) {
      setMessage('Please upload a CSV file first.');
      return;
    }

    const detectedRole = inferFrontendDatasetRole(file.name);

    const headerMap = new Map(csvHeaders.map((h) => [cleanHeader(h), h]));
    const inferredZomatoAgentId =
      headerMap.get('delivery_person_id') || headerMap.get('delivery_person_id_') || '';

    const inferredOrderDateCol = headerMap.get('order_date') || '';

    const effectiveMapping: FieldMapping = !autoDetectedReconstructed
      ? {
          ...mapping,
          customer_id: mapping.customer_id || mapping.order_id || '',
          agent_id:
            detectedRole === 'comparative_template'
              ? mapping.agent_id || inferredZomatoAgentId || ''
              : mapping.agent_id,
          order_date_col: mapping.order_date_col || inferredOrderDateCol || '',
        }
      : mapping;

    if (
      !effectiveMapping.depot_lat ||
      !effectiveMapping.depot_lon ||
      !effectiveMapping.customer_id ||
      !effectiveMapping.customer_lat ||
      !effectiveMapping.customer_lon
    ) {
      setMessage('Please complete the required field mapping.');
      return;
    }

    try {
      setBusy(true);
      setMessage('Validating dataset and building reconstructed baseline dataset context...');

      const result = await validateDataset(file, effectiveMapping, detectedRole);

      const datasetMeta: Dataset = {
        id: result.datasetId,
        name: file.name,
        rows: result.summary.records,
        depots: result.summary.depots,
        customers: result.summary.customers,
        orders: result.summary.orders,
        datasetRole: result.datasetRole,
        sourceLabel: result.sourceLabel,
        reconstructedBaselineReady: result.reconstructedBaselineReady,
        reconstructedBaselineName: result.reconstructedBaselineName,
      };

      localStorage.setItem(STORAGE_FILE_KEY, JSON.stringify(datasetMeta));
      localStorage.setItem(STORAGE_VALIDATION_KEY, JSON.stringify(result));

      setValidation(result);
      setMessage(
        `Validation completed successfully. ${result.sourceLabel} is ready, and the reconstructed baseline dataset has been prepared.`
      );
    } catch (err) {
      setMessage(err instanceof Error ? err.message : 'Validation failed.');
    } finally {
      setBusy(false);
    }
  };

  const selectOptions =
    csvHeaders.length > 0
      ? [
          { label: 'Select column', value: '' },
          ...csvHeaders.map((header) => ({
            label: header,
            value: header,
          })),
        ]
      : [{ label: 'Upload file first', value: '' }];

  return (
    <div className="space-y-6">
      <Card>
        <div className="space-y-4">
          <h2 className="text-lg font-semibold text-slate-900 mb-4">Dataset Upload</h2>
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-2">Upload CSV</label>
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="block w-full text-sm text-slate-700
                         file:mr-4 file:py-2 file:px-4
                         file:rounded-md file:border-0
                         file:text-sm file:font-semibold
                         file:bg-blue-600 file:text-white
                         hover:file:bg-blue-700"
            />
            {file && <p className="mt-2 text-sm text-slate-600">Selected file: {file.name}</p>}
            {inferredRole && (
              <p className="mt-1 text-xs text-slate-500">
                Detected dataset role: {roleBadgeText(inferredRole)}
              </p>
            )}
            {autoDetectedReconstructed && (
              <p className="mt-1 text-xs font-medium text-emerald-700">
                Standardized reconstructed schema detected. Manual mapping skipped.
              </p>
            )}
          </div>

          {!autoDetectedReconstructed && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {FIELD_OPTIONS.map((field) => (
                <Select
                  key={field.value}
                  label={field.label}
                  value={(mapping as Record<string, string | undefined>)[field.value] ?? ''}
                  onChange={(value) => setMapping((prev) => ({ ...prev, [field.value]: value }))}
                  options={selectOptions}
                />
              ))}
            </div>
          )}

          {autoDetectedReconstructed && (
            <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-800">
              The uploaded CSV already contains the reconstructed baseline schema. The system
              auto-filled the required fields:
              <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                <div>
                  <strong>Depot Latitude:</strong> {mapping.depot_lat}
                </div>
                <div>
                  <strong>Depot Longitude:</strong> {mapping.depot_lon}
                </div>
                <div>
                  <strong>Agent ID:</strong> {mapping.agent_id || 'Not detected'}
                </div>
                <div>
                  <strong>Customer Latitude:</strong> {mapping.customer_lat}
                </div>
                <div>
                  <strong>Customer Longitude:</strong> {mapping.customer_lon}
                </div>
                {mapping.order_id && (
                  <div>
                    <strong>Order ID:</strong> {mapping.order_id}
                  </div>
                )}
                {mapping.order_date_col && (
                  <div>
                    <strong>Order Date:</strong> {mapping.order_date_col}
                  </div>
                )}
                {mapping.eta_col && (
                  <div>
                    <strong>ETA Column:</strong> {mapping.eta_col}
                  </div>
                )}
                {mapping.rating_col && (
                  <div>
                    <strong>Rating Column:</strong> {mapping.rating_col}
                  </div>
                )}
                {mapping.area_col && (
                  <div>
                    <strong>Area/Cluster Column:</strong> {mapping.area_col}
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="flex gap-3 items-center flex-wrap">
            <Button onClick={onValidate} disabled={busy || backendReady === false}>
              {busy ? 'Processing...' : 'Validate Dataset'}
            </Button>
            <Button
              variant="secondary"
              onClick={() => navigate('/baseline')}
              disabled={!validation}
            >
              Proceed to Baseline
            </Button>

            {validation?.reconstructedBaselineReady && (
              <a
                href={getReconstructedDatasetDownloadUrl(validation.datasetId)}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
              >
                Download Reconstructed Baseline Dataset
              </a>
            )}
          </div>

          <div className="text-sm text-slate-600">
            Backend status:{' '}
            {backendReady === null ? 'checking...' : backendReady ? 'connected' : 'not reachable'}
          </div>

          {message && <div className="text-sm text-slate-700">{message}</div>}

          {validation && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 space-y-2">
              <div className="text-sm font-semibold text-slate-800">Validated Dataset Summary</div>
              <div className="text-sm text-slate-700">
                <strong>Source:</strong> {validation.sourceLabel}
              </div>
              <div className="text-sm text-slate-700">
                <strong>Role:</strong> {roleBadgeText(validation.datasetRole)}
              </div>
              <div className="text-sm text-slate-700">
                <strong>Reconstructed Baseline:</strong>{' '}
                {validation.reconstructedBaselineReady ? 'Ready' : 'Not ready'}
              </div>
              <div className="text-sm text-slate-700">
                <strong>Reconstructed File Name:</strong> {validation.reconstructedBaselineName}
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
                <div className="text-sm text-slate-700">
                  <strong>Records:</strong> {validation.summary.records}
                </div>
                <div className="text-sm text-slate-700">
                  <strong>Depots:</strong> {validation.summary.depots}
                </div>
                <div className="text-sm text-slate-700">
                  <strong>Customers:</strong> {validation.summary.customers}
                </div>
                <div className="text-sm text-slate-700">
                  <strong>Orders:</strong> {validation.summary.orders}
                </div>
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default DatasetUpload;
