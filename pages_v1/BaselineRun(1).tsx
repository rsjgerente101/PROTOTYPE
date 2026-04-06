import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '../src/components/Card';
import { Button } from '../src/components/Button';
import { Input } from '../src/components/Input';
import { Select } from '../src/components/Select';
import { KPICard } from '../src/components/KPICard';
import { MapCanvas } from '../src/components/MapCanvas';
import { RouteTable } from '../src/components/RouteTable';
import { MapIcon, ClockIcon, RouteIcon, UsersIcon, PlayIcon } from 'lucide-react';
import { MOCK_DATASET, generateBaselineRun } from '../utils/mockData.legacy';
import { AlgorithmRun, Depot } from '../src/types';

export function BaselineRun() {
  const navigate = useNavigate();

  const [parameters, setParameters] = useState({
    vehicles: '4',
    depot: 'D001',
    speed: '40',
    seed: '42'
  });

  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState<AlgorithmRun | null>(null);
  const [showLabels, setShowLabels] = useState(true);
  const [showRouteNumbers, setShowRouteNumbers] = useState(false);
  const [selectedRoute, setSelectedRoute] = useState<string | null>(null);

  const handleRun = () => {
    setIsRunning(true);

    setTimeout(() => {
      const baselineResult = generateBaselineRun();
      setResult(baselineResult);
      setSelectedRoute(baselineResult.routes[0]?.id || null);
      setIsRunning(false);
      localStorage.setItem('baselineRun', JSON.stringify(baselineResult));
    }, 2000);
  };

  const displayedRoute = result?.routes.find(r => r.id === selectedRoute);

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
            Back to Upload
          </Button>
        </div>

        <div className="flex-1 flex overflow-hidden">

          {/* Map */}
          <div className="flex-1 p-6 overflow-auto" style={{ minWidth: 0 }}>
            {result ? (
              <MapCanvas
                routes={result.routes}
                depot={MOCK_DATASET.depots[0]}
                showLabels={showLabels}
                showRouteNumbers={showRouteNumbers}
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


          <div className="w-96 bg-white border-l border-gray-200 p-6 overflow-auto">

            {/* Parameters */}
            <Card className="mb-6">
              <h2 className="text-sm font-semibold text-gray-900 mb-4">
                Parameters
              </h2>

              <div className="space-y-3">
                <Input
                  label="Number of Vehicles"
                  type="number"
                  value={parameters.vehicles}
                  onChange={val =>
                    setParameters({ ...parameters, vehicles: val })
                  }
                />

                <Select
                  label="Depot"
                  value={parameters.depot}
                  onChange={val =>
                    setParameters({ ...parameters, depot: val })
                  }
                  options={MOCK_DATASET.depots.map((d: Depot) => ({
                    value: d.id,
                    label: d.name ?? d.id
                  }))}
                />

                <Input
                  label="Speed Factor (km/h)"
                  type="number"
                  value={parameters.speed}
                  onChange={val =>
                    setParameters({ ...parameters, speed: val })
                  }
                />

                <Input
                  label="Random Seed"
                  type="number"
                  value={parameters.seed}
                  onChange={val =>
                    setParameters({ ...parameters, seed: val })
                  }
                />
              </div>

              <Button
                onClick={handleRun}
                disabled={isRunning}
                className="w-full mt-4 flex items-center justify-center gap-2"
              >
                {isRunning ? (
                  <span>Processing...</span>
                ) : (
                  <>
                    <PlayIcon className="w-4 h-4" />
                    <span>Run Baseline Algorithm</span>
                  </>
                )}
              </Button>
            </Card>

            {/* Results */}
            {result && (
              <>
                <div className="mb-6">
                  <h2 className="text-sm font-semibold text-gray-900 mb-3">
                    Key Performance Indicators
                  </h2>
                  <div className="space-y-3">
                    <KPICard
                      title="Total Distance"
                      value={result.kpis.totalDistance.toFixed(2)}
                      unit="km"
                      icon={<RouteIcon className="w-5 h-5" />}
                    />
                    <KPICard
                      title="Total Time"
                      value={result.kpis.totalTime.toFixed(2)}
                      unit="hrs"
                      icon={<ClockIcon className="w-5 h-5" />}
                    />
                    <KPICard
                      title="Compute Time"
                      value={result.kpis.computeTime}
                      unit="ms"
                    />
                    <KPICard
                      title="Number of Stops"
                      value={result.kpis.numberOfStops}
                      icon={<UsersIcon className="w-5 h-5" />}
                    />
                  </div>
                </div>

                <div className="mb-6">
                  <h2 className="text-sm font-semibold text-gray-900 mb-3">
                    Map Controls
                  </h2>
                  <div className="space-y-2">
                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={showLabels}
                        onChange={e => setShowLabels(e.target.checked)}
                      />
                      Show Customer Labels
                    </label>

                    <label className="flex items-center gap-2 text-sm">
                      <input
                        type="checkbox"
                        checked={showRouteNumbers}
                        onChange={e => setShowRouteNumbers(e.target.checked)}
                      />
                      Show Route Numbers
                    </label>
                  </div>
                </div>

                <div className="mb-6">
                  <Select
                    label="Select Route"
                    value={selectedRoute || ''}
                    onChange={setSelectedRoute}
                    options={result.routes.map(r => ({
                      value: r.id,
                      label: r.representativeName
                    }))}
                  />
                </div>

                {displayedRoute && (
                  <RouteTable
                    routes={[displayedRoute]}
                    title={`Route: ${displayedRoute.representativeName}`}
                  />
                )}

                <Button
                  onClick={() => navigate('/enhanced')}
                  className="w-full mt-6"
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
}