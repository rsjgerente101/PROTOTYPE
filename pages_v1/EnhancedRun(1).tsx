import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '../src/components/Card';
import { Button } from '../src/components/Button';
import { KPICard } from '../src/components/KPICard';
import { MapCanvas } from '../src/components/MapCanvas';
import { DequePanel } from '../src/components/DequePanel';
import { PlayIcon } from 'lucide-react';
import { MOCK_DATASET, generateEnhancedRun } from '../utils/mockData.legacy';
import { AlgorithmRun, Representative } from '../src/types';

export function EnhancedRun() {
  const navigate = useNavigate();

  const [baselineRun, setBaselineRun] = useState<AlgorithmRun | null>(null);
  const [enhancedRun, setEnhancedRun] = useState<AlgorithmRun | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [representatives, setRepresentatives] = useState<Representative[]>([]);
  const [highlightedNodes, setHighlightedNodes] = useState<string[]>([]);
  const sidebarOpen = true;

  useEffect(() => {
    const stored = localStorage.getItem('baselineRun');
    if (stored) {
      setBaselineRun(JSON.parse(stored));
    } else {
      navigate('/baseline');
    }
  }, [navigate]);

  const handleRun = () => {
    if (!baselineRun) return;
    setIsRunning(true);
    setTimeout(() => {
      const enhanced = generateEnhancedRun(baselineRun);
      setEnhancedRun(enhanced);
      setRepresentatives(enhanced.representatives);
      setHighlightedNodes(['C007', 'C012']);
      setIsRunning(false);
      localStorage.setItem('enhancedRun', JSON.stringify(enhanced));
    }, 2500);
  };

  const handleUpdateRepresentatives = (updatedReps: Representative[]) => {
    setRepresentatives(updatedReps);
  };

  const calculateDelta = (enhanced: number, baseline: number): number => {
    return ((enhanced - baseline) / baseline) * 100;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="h-screen flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 px-8 py-4 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">
              Enhanced Route Optimization
            </h1>
            <p className="text-sm text-gray-600">
              Greedy Nearest Neighbor + Double-Ended Queue Rebalancing
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button variant="outline" onClick={() => navigate('/baseline')}>
              Back to Baseline
            </Button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Map */}
          <div
            className={`flex-1 p-6 overflow-auto transition-all duration-300 ${sidebarOpen ? '' : 'w-full'
              }`}
            style={{ minWidth: 0 }}
          >
            {enhancedRun ? (
              <MapCanvas
                routes={enhancedRun.routes}
                depot={MOCK_DATASET.depots[0]}
                showLabels
                showRouteNumbers
                highlightedNodes={highlightedNodes}
              />
            ) : (
              <Card className="h-full flex items-center justify-center">
                <div className="text-center flex flex-col items-center justify-center gap-4">
                  <p className="text-gray-500 text-lg font-medium">
                    Run the enhanced algorithm with DEQ rebalancing
                  </p>
                  <Button
                    onClick={handleRun}
                    disabled={isRunning}
                    className="flex items-center justify-center gap-2"
                  >
                    {isRunning ? (
                      <span>Processing...</span>
                    ) : (
                      <>
                        <PlayIcon className="w-4 h-4" />
                        <span>Run Enhanced Algorithm</span>
                      </>
                    )}
                  </Button>
                </div>
              </Card>
            )}
          </div>

          {/* Sidebar */}
          {sidebarOpen && (
            <div className="w-96 bg-white border-l border-gray-200 p-6 overflow-auto transition-all duration-300">
              {!enhancedRun ? (
                <Card>
                  <h2 className="text-sm font-semibold text-gray-900 mb-4">
                    Enhanced Algorithm
                  </h2>
                  <p className="text-sm text-gray-600 mb-4">
                    This algorithm applies Double-Ended Queue rebalancing to optimize workload distribution and improve fairness metrics.
                  </p>
                  <Button
                    onClick={handleRun}
                    disabled={isRunning}
                    className="w-full flex items-center justify-center gap-2"
                  >
                    {isRunning ? (
                      <span>Processing...</span>
                    ) : (
                      <>
                        <PlayIcon className="w-4 h-4" />
                        <span>Run Enhanced Algorithm</span>
                      </>
                    )}
                  </Button>
                </Card>
              ) : (
                <>
                  {/* KPIs */}
                  <div className="mb-6">
                    <h2 className="text-sm font-semibold text-gray-900 mb-3">
                      Δ-KPI (vs Baseline)
                    </h2>
                    <div className="space-y-3">
                      <KPICard
                        title="Δ Distance"
                        value={enhancedRun.kpis.totalDistance.toFixed(2)}
                        unit="km"
                        delta={
                          baselineRun
                            ? calculateDelta(
                              enhancedRun.kpis.totalDistance,
                              baselineRun.kpis.totalDistance
                            )
                            : undefined
                        }
                        deltaLabel="vs baseline"
                      />
                      <KPICard
                        title="Δ Time"
                        value={enhancedRun.kpis.totalTime.toFixed(2)}
                        unit="hrs"
                        delta={
                          baselineRun
                            ? calculateDelta(
                              enhancedRun.kpis.totalTime,
                              baselineRun.kpis.totalTime
                            )
                            : undefined
                        }
                        deltaLabel="vs baseline"
                      />
                      <KPICard
                        title="Δ Coverage"
                        value={(enhancedRun.kpis.coverageRatio! * 100).toFixed(1)}
                        unit="%"
                        delta={
                          baselineRun?.kpis.coverageRatio
                            ? calculateDelta(
                              enhancedRun.kpis.coverageRatio!,
                              baselineRun.kpis.coverageRatio
                            )
                            : undefined
                        }
                        deltaLabel="vs baseline"
                      />
                      <KPICard
                        title="Δ Fairness (WBI)"
                        value={enhancedRun.kpis.workloadBalanceIndex!.toFixed(2)}
                        delta={
                          baselineRun?.kpis.workloadBalanceIndex
                            ? calculateDelta(
                              enhancedRun.kpis.workloadBalanceIndex!,
                              baselineRun.kpis.workloadBalanceIndex
                            )
                            : undefined
                        }
                        deltaLabel="vs baseline"
                      />
                    </div>
                  </div>

                  <div className="mb-6">
                    <DequePanel
                      representatives={representatives}
                      onUpdate={handleUpdateRepresentatives}
                    />
                  </div>

                  <Button
                    onClick={() => navigate('/compare')}
                    className="w-full flex items-center justify-center"
                  >
                    Proceed to Comparison View
                  </Button>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
