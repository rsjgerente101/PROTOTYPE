import React, { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '../components/Card';
import { Button } from '../components/Button';
import { ComparisonTile } from '../components/ComparisonTile';
import { FileTextIcon, TableIcon, ArrowLeftIcon } from 'lucide-react';
import type { AlgorithmRun } from '../types';

type StoredRunSummary = Pick<
  AlgorithmRun,
  'id' | 'algorithm' | 'datasetId' | 'kpis' | 'representatives'
>;

const BASELINE_STORAGE_KEY = 'baselineRunSummary';
const ENHANCED_STORAGE_KEY = 'enhancedRunSummary';

const CompareExport: React.FC = () => {
  const navigate = useNavigate();

  const [baselineRun, setBaselineRun] = useState<StoredRunSummary | null>(null);
  const [enhancedRun, setEnhancedRun] = useState<StoredRunSummary | null>(null);

  useEffect(() => {
    const baseline = localStorage.getItem(BASELINE_STORAGE_KEY);
    const enhanced = localStorage.getItem(ENHANCED_STORAGE_KEY);

    if (baseline) setBaselineRun(JSON.parse(baseline));
    if (enhanced) setEnhancedRun(JSON.parse(enhanced));
  }, []);

  const comparisons = useMemo(() => {
    if (!baselineRun || !enhancedRun) return [];

    return [
      {
        label: 'Total Distance (km)',
        baseline: baselineRun.kpis.totalDistance,
        enhanced: enhancedRun.kpis.totalDistance,
        lowerIsBetter: true,
      },
      {
        label: 'Travel Time (min)',
        baseline: baselineRun.kpis.travelTime,
        enhanced: enhancedRun.kpis.travelTime,
        lowerIsBetter: true,
      },
      {
        label: 'Operational Time (min)',
        baseline: baselineRun.kpis.operationalTime,
        enhanced: enhancedRun.kpis.operationalTime,
        lowerIsBetter: true,
      },
      {
        label: 'Fairness',
        baseline: baselineRun.kpis.fairness,
        enhanced: enhancedRun.kpis.fairness,
        lowerIsBetter: false,
      },
      {
        label: 'Workload Balance',
        baseline: baselineRun.kpis.workloadBalance,
        enhanced: enhancedRun.kpis.workloadBalance,
        lowerIsBetter: false,
      },
      {
        label: 'Scalability',
        baseline: baselineRun.kpis.scalability,
        enhanced: enhancedRun.kpis.scalability,
        lowerIsBetter: false,
      },
    ];
  }, [baselineRun, enhancedRun]);

  const handleExportCSV = () => {
    if (!baselineRun || !enhancedRun) return;

    const rows = [
      ['Metric', 'Baseline', 'Enhanced'],
      ['Total Distance (km)', baselineRun.kpis.totalDistance, enhancedRun.kpis.totalDistance],
      ['Travel Time (min)', baselineRun.kpis.travelTime, enhancedRun.kpis.travelTime],
      ['Operational Time (min)', baselineRun.kpis.operationalTime, enhancedRun.kpis.operationalTime],
      ['Fairness', baselineRun.kpis.fairness, enhancedRun.kpis.fairness],
      ['Workload Balance', baselineRun.kpis.workloadBalance, enhancedRun.kpis.workloadBalance],
      ['Scalability', baselineRun.kpis.scalability, enhancedRun.kpis.scalability],
    ];

    const csvContent = rows.map((row) => row.join(',')).join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'baseline_vs_enhanced_comparison.csv';
    a.click();

    window.URL.revokeObjectURL(url);
  };

  const handleExportJSON = () => {
    if (!baselineRun || !enhancedRun) return;

    const payload = {
      exportedAt: new Date().toISOString(),
      baseline: baselineRun,
      enhanced: enhancedRun,
      comparison: comparisons,
    };

    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    });
    const url = window.URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = 'baseline_vs_enhanced_comparison.json';
    a.click();

    window.URL.revokeObjectURL(url);
  };

  if (!baselineRun || !enhancedRun) {
    return (
      <Card title="Compare & Export">
        Baseline and enhanced run summaries are required.
      </Card>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
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
              <span className="inline-flex items-center gap-2">
                <ArrowLeftIcon className="w-4 h-4" />
                Back to Enhanced
              </span>
            </Button>
          </div>
        </div>

        <div className="flex-1 p-8 overflow-auto">
          <div className="max-w-6xl mx-auto space-y-8">
            <Card title="Baseline vs Enhanced Performance">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {comparisons.map((item) => (
                  <ComparisonTile
                    key={item.label}
                    label={item.label}
                    baselineValue={item.baseline}
                    enhancedValue={item.enhanced}
                    lowerIsBetter={item.lowerIsBetter}
                  />
                ))}
              </div>
            </Card>

            <Card title="Export Options">
              <div className="flex flex-wrap gap-4">
                <Button
                  onClick={handleExportCSV}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  <TableIcon className="w-4 h-4" />
                  Export Comparison CSV
                </Button>

                <Button
                  onClick={handleExportJSON}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  <FileTextIcon className="w-4 h-4" />
                  Export Comparison JSON
                </Button>
              </div>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CompareExport;