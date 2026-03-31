import React, { useEffect, useMemo, useState } from 'react';
import { Card } from '../components/Card';
import { ComparisonTile } from '../components/ComparisonTile';
import type { AlgorithmRun } from '../types';

type StoredRunSummary = Pick<
  AlgorithmRun,
  'id' | 'algorithm' | 'datasetId' | 'kpis' | 'representatives'
>;

const BASELINE_STORAGE_KEY = 'baselineRunSummary';
const ENHANCED_STORAGE_KEY = 'enhancedRunSummary';

const CompareExport: React.FC = () => {
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
        label: 'Coverage (%)',
        baseline: baselineRun.kpis.coverage,
        enhanced: enhancedRun.kpis.coverage,
        lowerIsBetter: false,
      },
      {
        label: 'Scalability',
        baseline: baselineRun.kpis.scalability,
        enhanced: enhancedRun.kpis.scalability,
        lowerIsBetter: false,
      },
      {
        label: 'Compute Time',
        baseline: baselineRun.kpis.computeTime,
        enhanced: enhancedRun.kpis.computeTime,
        lowerIsBetter: true,
      },
    ];
  }, [baselineRun, enhancedRun]);

  if (!baselineRun || !enhancedRun) {
    return (
      <Card title="Compare & Export">
        Baseline and enhanced run summaries are required.
      </Card>
    );
  }

  return (
    <div className="space-y-6">
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
    </div>
  );
};

export default CompareExport;