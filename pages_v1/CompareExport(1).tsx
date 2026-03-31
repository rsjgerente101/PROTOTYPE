import React, { useEffect, useState, createElement } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '../src/components/Card';
import { Button } from '../src/components/Button';
import { ComparisonTile } from '../src/components/ComparisonTile';
import { FileTextIcon, TableIcon } from 'lucide-react';
import { AlgorithmRun } from '../src/types';
export function CompareExport() {
  const navigate = useNavigate();
  const [baselineRun, setBaselineRun] = useState<AlgorithmRun | null>(null);
  const [enhancedRun, setEnhancedRun] = useState<AlgorithmRun | null>(null);
  useEffect(() => {
    const baseline = localStorage.getItem('baselineRun');
    const enhanced = localStorage.getItem('enhancedRun');
    if (baseline) setBaselineRun(JSON.parse(baseline));
    if (enhanced) setEnhancedRun(JSON.parse(enhanced));
    if (!baseline || !enhanced) {
      navigate('/baseline');
    }
  }, [navigate]);
  const handleExportCSV = () => {
    const csvContent = `Algorithm,Total Distance (km),Total Time (hrs),Compute Time (ms),Delay Score,Rating Penalty
Baseline,${baselineRun?.kpis.totalDistance.toFixed(2)},${baselineRun?.kpis.totalTime.toFixed(2)},${baselineRun?.kpis.computeTime},${baselineRun?.kpis.delayScore?.toFixed(2)},${baselineRun?.kpis.ratingPenalty?.toFixed(2)}
Enhanced,${enhancedRun?.kpis.totalDistance.toFixed(2)},${enhancedRun?.kpis.totalTime.toFixed(2)},${enhancedRun?.kpis.computeTime},${enhancedRun?.kpis.delayScore?.toFixed(2)},${enhancedRun?.kpis.ratingPenalty?.toFixed(2)}`;
    const blob = new Blob([csvContent], {
      type: 'text/csv'
    });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'route_comparison.csv';
    a.click();
  };
  const handleExportJSON = () => {
    const jsonContent = JSON.stringify({
      baseline: baselineRun,
      enhanced: enhancedRun,
      comparison: {
        timestamp: new Date().toISOString()
      }
    }, null, 2);
    const blob = new Blob([jsonContent], {
      type: 'application/json'
    });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'run_log.json';
    a.click();
  };
  if (!baselineRun || !enhancedRun) {
    return <div>Loading...</div>;
  }
  return <div className="min-h-screen bg-gray-50">
      <div className="h-screen flex flex-col">
        <div className="bg-white border-b border-gray-200 px-8 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Compare Results & Export
              </h1>
              <p className="text-sm text-gray-600">
                Baseline vs Enhanced Algorithm Performance
              </p>
            </div>
            <Button variant="outline" onClick={() => navigate('/enhanced')}>
              Back to Enhanced
            </Button>
          </div>
        </div>

        <div className="flex-1 p-8 overflow-auto">
          <div className="max-w-6xl mx-auto space-y-8">
            <div>
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Side-by-Side Comparison
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                <ComparisonTile title="Total Distance" baselineValue={baselineRun.kpis.totalDistance} enhancedValue={enhancedRun.kpis.totalDistance} unit="km" lowerIsBetter={true} />
                <ComparisonTile title="Total Time" baselineValue={baselineRun.kpis.totalTime} enhancedValue={enhancedRun.kpis.totalTime} unit="hrs" lowerIsBetter={true} />
                <ComparisonTile title="Compute Time" baselineValue={baselineRun.kpis.computeTime} enhancedValue={enhancedRun.kpis.computeTime} unit="ms" lowerIsBetter={true} />
                <ComparisonTile title="Delay Score" baselineValue={baselineRun.kpis.delayScore!} enhancedValue={enhancedRun.kpis.delayScore!} lowerIsBetter={true} />
                <ComparisonTile title="Rating Penalty" baselineValue={baselineRun.kpis.ratingPenalty!} enhancedValue={enhancedRun.kpis.ratingPenalty!} lowerIsBetter={true} />
              </div>
            </div>

            <Card>
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Export Options
              </h2>
              <div className="flex gap-4">
                <Button onClick={handleExportCSV} variant="outline" className="flex items-center gap-2">
                  <TableIcon className="w-4 h-4" />
                  Export Routes (CSV)
                </Button>
                <Button onClick={handleExportJSON} variant="outline" className="flex items-center gap-2">
                  <FileTextIcon className="w-4 h-4" />
                  Export Run Log (JSON)
                </Button>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>;
}