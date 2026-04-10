import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowLeftIcon,
  ClockIcon,
  MapIcon,
  PlayIcon,
  RouteIcon,
  UsersIcon,
} from 'lucide-react';

import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { Input } from '../components/Input';
import { KPICard } from '../components/KPICard';
import { MapCanvas } from '../components/MapCanvas';
import { RouteTable } from '../components/RouteTable';
import { Select } from '../components/Select';

import type { AlgorithmRun, Dataset, Depot, Route } from '../types';
import { getDatasetMeta, runBaseline } from '../services/deliveryApi';

type StoredRunSummary = {
  id: string;
  algorithm: string;
  datasetId: string;
  kpis: AlgorithmRun['kpis'];
  representatives: AlgorithmRun['representatives'];
};

const BASELINE_STORAGE_KEY = 'baselineRunSummary';
const DATASET_STORAGE_KEY = 'uploadedDatasetFileMeta';

const BaselineRun: React.FC = () => {
  const navigate = useNavigate();

  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [depot, setDepot] = useState<Depot | null>(null);
  const [run, setRun] = useState<AlgorithmRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');

  const [parameters, setParameters] = useState({
    vehicles: '4',
    speed: '18.75',
    serviceMinutes: '8',
    seed: '42',
  });

  const [showLabels, setShowLabels] = useState(true);
  const [showRouteNumbers, setShowRouteNumbers] = useState(false);
  const [selectedRouteId, setSelectedRouteId] = useState('');
  const [selectedStopNodeId, setSelectedStopNodeId] = useState('');
  const [showAllRoutes, setShowAllRoutes] = useState(false);

  useEffect(() => {
    const storedDataset = localStorage.getItem(DATASET_STORAGE_KEY);
    if (storedDataset) {
      const parsed = JSON.parse(storedDataset);
      setDataset(parsed);
    }
  }, []);

  useEffect(() => {
    const loadMeta = async () => {
      if (!dataset?.id) return;
      try {
        const meta = await getDatasetMeta(dataset.id);
        if (meta.depot) {
          setDepot(meta.depot);
        }
      } catch {
        // keep silent for now
      }
    };

    loadMeta();
  }, [dataset]);

  const selectedRoute: Route | null = useMemo(() => {
    if (!run || !selectedRouteId) return null;
    return run.routes.find((route) => route.id === selectedRouteId) ?? null;
  }, [run, selectedRouteId]);

  const selectedRoutesForDisplay = useMemo(() => {
    if (!run) return [];
    if (showAllRoutes) return run.routes;
    if (!selectedRouteId) return run.routes;
    const match = run.routes.find((route) => route.id === selectedRouteId);
    return match ? [match] : run.routes;
  }, [run, selectedRouteId, showAllRoutes]);

  const displayDepot: Depot | null = useMemo(() => {
    if (run?.previewSummary?.depotLat != null && run?.previewSummary?.depotLon != null) {
      return {
        id: 'PREVIEW-DEPOT',
        name: 'Depot',
        lat: run.previewSummary.depotLat,
        lon: run.previewSummary.depotLon,
      };
    }
    return depot;
  }, [run, depot]);

  const selectedStopOptions = useMemo(() => {
    if (!selectedRoute) return [];
    return selectedRoute.stops.map((stop) => ({
      value: stop.nodeId,
      label: `Stop ${stop.stopNumber} - ${stop.nodeName}`,
    }));
  }, [selectedRoute]);

  useEffect(() => {
    if (selectedRoute?.stops?.length) {
      setSelectedStopNodeId(selectedRoute.stops[0].nodeId);
    } else {
      setSelectedStopNodeId('');
    }
  }, [selectedRoute]);

  const handleRun = async () => {
    if (!dataset) {
      setMessage('No validated dataset found. Please upload and validate first.');
      return;
    }

    try {
      setBusy(true);
      setMessage('Running baseline experiment on backend...');

      const result = await runBaseline({
        datasetId: dataset.id,
        numRepresentatives: Number(parameters.vehicles) || 4,
        avgSpeedKmph: Number(parameters.speed) || 18.75,
        serviceMinutesPerStop: Number(parameters.serviceMinutes) || 8,
        seed: Number(parameters.seed) || 42,
        runProfile:
          dataset.datasetRole === 'primary_reconstruction'
            ? 'amazon_expanded_search'
            : 'default_balanced',
      });

      const summary: StoredRunSummary = {
        id: result.id,
        algorithm: result.algorithm,
        datasetId: result.datasetId,
        kpis: result.kpis,
        representatives: result.representatives,
      };

      localStorage.setItem(BASELINE_STORAGE_KEY, JSON.stringify(summary));
      setRun(result);
      setSelectedRouteId(result.routes[0]?.id ?? '');
      setMessage('Baseline run completed.');
    } catch (err) {
      setMessage(err instanceof Error ? err.message : 'Baseline run failed.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="h-screen flex flex-col">
        <div className="bg-white border-b border-gray-200 px-8 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Baseline Route Generation
            </h1>
            <p className="text-sm text-gray-600">
              Greedy Nearest Neighbor + Dijkstra Shortest Path
            </p>
          </div>

          <Button variant="outline" onClick={() => navigate('/')}>
            <span className="inline-flex items-center gap-2">
              <ArrowLeftIcon className="w-4 h-4" />
              Back to Upload
            </span>
          </Button>
        </div>

        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 p-6 overflow-auto" style={{ minWidth: 0 }}>
            {run ? (
              <MapCanvas
                routes={selectedRoutesForDisplay}
                depot={displayDepot}
                showLabels={showLabels}
                showRouteNumbers={showRouteNumbers}
                highlightedNodes={selectedStopNodeId ? [selectedStopNodeId] : []}
              />
            ) : (
              <Card className="h-full flex items-center justify-center">
                <div className="text-center">
                  <MapIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
                  <p className="text-gray-500">
                    Run the baseline algorithm to visualize routes
                  </p>
                </div>
              </Card>
            )}
          </div>

          <div className="w-[420px] bg-white border-l border-gray-200 p-6 overflow-auto">
            <Card className="mb-6">
              <h2 className="text-sm font-semibold text-gray-900 mb-4">
                Parameters
              </h2>

              <div className="space-y-3">
                <Input
                  label="Number of Vehicles"
                  type="number"
                  value={parameters.vehicles}
                  onChange={(val) =>
                    setParameters((prev) => ({ ...prev, vehicles: val }))
                  }
                  disabled
                />

                <Input
                  label="Average Speed (km/h)"
                  type="number"
                  value={parameters.speed}
                  onChange={(val) =>
                    setParameters((prev) => ({ ...prev, speed: val }))
                  }
                  disabled
                />

                <Input
                  label="Service Minutes per Stop"
                  type="number"
                  value={parameters.serviceMinutes}
                  onChange={(val) =>
                    setParameters((prev) => ({ ...prev, serviceMinutes: val }))
                  }
                  disabled
                />

                {/* Random seed is kept internal for reproducibility and hidden from the UI */}
              </div>

              <Button
                onClick={handleRun}
                disabled={busy}
                className="w-full mt-4 flex items-center justify-center gap-2"
              >
                {busy ? (
                  <span>Processing...</span>
                ) : (
                  <>
                    <PlayIcon className="w-4 h-4" />
                    <span>Run Baseline Algorithm</span>
                  </>
                )}
              </Button>

              {message && (
                <div className="mt-3 text-sm text-slate-700">{message}</div>
              )}
            </Card>

            {run && (
              <>
                <div className="mb-6">
                  <h2 className="text-sm font-semibold text-gray-900 mb-3">
                    Key Performance Indicators
                  </h2>

                  <div className="space-y-3">
                    <KPICard title="Total Distance" value={run.kpis.totalDistance} unit="km" icon={<RouteIcon className="w-5 h-5" />} />
                    <KPICard title="Travel Time" value={Number((run.kpis.travelTime / 60).toFixed(2))} unit="hr" icon={<ClockIcon className="w-5 h-5" />} />
                    <KPICard title="Operational Time" value={Number((run.kpis.operationalTime / 60).toFixed(2))} unit="hr" icon={<ClockIcon className="w-5 h-5" />} />
                    <KPICard title="Fairness" value={run.kpis.fairness} icon={<UsersIcon className="w-5 h-5" />} />
                  </div>
                </div>

                <div className="mb-6">
                  <h2 className="text-sm font-semibold text-gray-900 mb-3">
                    Map Controls
                  </h2>

                  <div className="space-y-2">
                    <label className="flex items-center gap-2 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        checked={showLabels}
                        onChange={(e) => setShowLabels(e.target.checked)}
                      />
                      Show Customer Labels
                    </label>

                    <label className="flex items-center gap-2 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        checked={showRouteNumbers}
                        onChange={(e) => setShowRouteNumbers(e.target.checked)}
                      />
                      Show Route Numbers
                    </label>

                    <label className="flex items-center gap-2 text-sm text-gray-700">
                      <input
                        type="checkbox"
                        checked={showAllRoutes}
                        onChange={(e) => setShowAllRoutes(e.target.checked)}
                      />
                      Show All Representatives
                    </label>
                  </div>
                </div>

                <div className="mb-6">
                  <Select
                    label="Select Route"
                    value={selectedRouteId}
                    onChange={setSelectedRouteId}
                    options={run.routes.map((route) => ({
                      value: route.id,
                      label: route.representativeName,
                    }))}
                  />
                </div>

                <div className="mb-6">
                  <Select
                    label="Select Stop"
                    value={selectedStopNodeId}
                    onChange={setSelectedStopNodeId}
                    options={selectedStopOptions}
                  />
                </div>

                <RouteTable
                  routes={selectedRoute ? [selectedRoute] : []}
                  title={
                    selectedRoute
                      ? `Route: ${selectedRoute.representativeName}`
                      : 'Selected Route'
                  }
                />

                <Button
                  onClick={() => navigate('/enhanced')}
                  variant="secondary"
                  className="w-full mt-6"
                  disabled={!run}
                >
                  Use Baseline as Seed for Enhanced Model
                </Button>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default BaselineRun;