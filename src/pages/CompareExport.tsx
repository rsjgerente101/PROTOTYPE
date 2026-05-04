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

const formatSalesRepName = (repId?: string | null) => {
  if (!repId) return '';
  return repId.replace('-AGE-', '-');
};

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
        label: 'Average Total Distance (km)',
        baseline: baselineRun.kpis.avgTotalDistance,
        enhanced: enhancedRun.kpis.avgTotalDistance,
        lowerIsBetter: true,
      },
      {
        label: 'Average Travel Time (hr)',
        baseline: baselineRun.kpis.avgTravelTime,
        enhanced: enhancedRun.kpis.avgTravelTime,
        lowerIsBetter: true,
      },
      {
        label: 'Operational Time (hr)',
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
        label: 'Workload Balance (%)',
        baseline: baselineRun.kpis.workloadBalance,
        enhanced: enhancedRun.kpis.workloadBalance,
        lowerIsBetter: true,
      },
      {
        label: 'Coverage Ratio (%)',
        baseline: baselineRun.kpis.coverageRatio,
        enhanced: enhancedRun.kpis.coverageRatio,
        lowerIsBetter: false,
      },
    ];
  }, [baselineRun, enhancedRun]);

  const handleExportCSV = () => {
    if (!baselineRun || !enhancedRun) return;

    const rows = [
      ['Metric', 'Baseline', 'Enhanced'],
      ['Average Total Distance (km)', baselineRun.kpis.avgTotalDistance, enhancedRun.kpis.avgTotalDistance],
      ['Average Travel Time (hr)', baselineRun.kpis.avgTravelTime, enhancedRun.kpis.avgTravelTime],
      ['Operational Time (hr)', baselineRun.kpis.operationalTime, enhancedRun.kpis.operationalTime],
      ['Fairness', baselineRun.kpis.fairness, enhancedRun.kpis.fairness],
      ['Workload Balance (%)', baselineRun.kpis.workloadBalance, enhancedRun.kpis.workloadBalance],
      ['Coverage Ratio (%)', baselineRun.kpis.coverageRatio, enhancedRun.kpis.coverageRatio],
    ];

    rows.push([]);
    rows.push([
      'Per Sales Rep Comparison Report',
      '',
      '',
      '',
      '',
      '',
      '',
    ]);
    rows.push([
      'Sales Rep',
      'Baseline Customers',
      'Enhanced Customers',
      'Baseline Workload',
      'Enhanced Workload',
      'Baseline Distance (km)',
      'Enhanced Distance (km)',
    ]);

    repComparisonRows.forEach((row) => {
      rows.push([
        formatSalesRepName(row.repId),
        row.baselineCustomers,
        row.enhancedCustomers,
        row.baselineWorkload.toFixed(2),
        row.enhancedWorkload.toFixed(2),
        row.baselineDistance.toFixed(2),
        row.enhancedDistance.toFixed(2),
      ]);
    });

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
      repComparisonReport: repComparisonRows,
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

  const buildRepComparisonRows = () => {
    const baselineMap = new Map(
      baselineRun.representatives.map((rep) => [rep.id, rep])
    );
    const enhancedMap = new Map(
      enhancedRun.representatives.map((rep) => [rep.id, rep])
    );

    const allRepIds = Array.from(
      new Set([
        ...baselineRun.representatives.map((rep) => rep.id),
        ...enhancedRun.representatives.map((rep) => rep.id),
      ])
    ).sort();


    return allRepIds.map((repId) => {
      const baselineRep = baselineMap.get(repId);
      const enhancedRep = enhancedMap.get(repId);

      const baselineCustomers = baselineRep?.assignedCustomers ?? 0;
      const enhancedCustomers = enhancedRep?.assignedCustomers ?? 0;

      const baselineWorkload = baselineRep?.workload ?? 0;
      const enhancedWorkload = enhancedRep?.workload ?? 0;

      const baselineDistance = baselineRep?.totalDistance ?? 0;
      const enhancedDistance = enhancedRep?.totalDistance ?? 0;

      return {
        repId,
        baselineCustomers,
        enhancedCustomers,
        baselineWorkload,
        enhancedWorkload,
        baselineDistance,
        enhancedDistance,
      };
    });
  };

  const repComparisonRows = buildRepComparisonRows();

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

            <div className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm">
              <div className="mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                  Per Sales Rep Comparison Report
                </h3>
                <p className="text-sm text-gray-500">
                  Baseline vs Enhanced comparison by sales representative.
                </p>
              </div>

              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left font-medium text-gray-700">Sales Rep</th>
                      <th className="px-4 py-3 text-right font-medium text-gray-700">Baseline Customers</th>
                      <th className="px-4 py-3 text-right font-medium text-gray-700">Enhanced Customers</th>
                      <th className="px-4 py-3 text-right font-medium text-gray-700">Baseline Workload</th>
                      <th className="px-4 py-3 text-right font-medium text-gray-700">Enhanced Workload</th>
                      <th className="px-4 py-3 text-right font-medium text-gray-700">Baseline Distance</th>
                      <th className="px-4 py-3 text-right font-medium text-gray-700">Enhanced Distance</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-200">
                    {repComparisonRows.map((row) => (
                      <tr key={row.repId} className="hover:bg-gray-50">
                        <td className="px-4 py-3 text-gray-900 font-medium">{formatSalesRepName(row.repId)}</td>
                        <td className="px-4 py-3 text-right text-gray-700">{row.baselineCustomers}</td>
                        <td className="px-4 py-3 text-right text-gray-700">{row.enhancedCustomers}</td>
                        <td className="px-4 py-3 text-right text-gray-700">{row.baselineWorkload.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right text-gray-700">{row.enhancedWorkload.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right text-gray-700">{row.baselineDistance.toFixed(2)}km</td>
                        <td className="px-4 py-3 text-right text-gray-700">{row.enhancedDistance.toFixed(2)}km</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

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