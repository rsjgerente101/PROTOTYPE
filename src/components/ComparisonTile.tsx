import React from 'react';
import { Card } from './Card';

interface ComparisonTileProps {
  label?: string;
  title?: string;
  baselineValue: number;
  enhancedValue: number;
  unit?: string;
  lowerIsBetter?: boolean;
}

export function ComparisonTile({
  label,
  title,
  baselineValue,
  enhancedValue,
  unit = '',
  lowerIsBetter = false,
}: ComparisonTileProps) {
  const displayLabel = label ?? title ?? 'Metric';
  const safeBaseline = baselineValue === 0 ? 1 : baselineValue;
  const delta = ((enhancedValue - baselineValue) / safeBaseline) * 100;
  const isImprovement = lowerIsBetter ? delta < 0 : delta > 0;
  const deltaColor = isImprovement ? 'text-green-600' : 'text-red-600';
  const deltaBgColor = isImprovement ? 'bg-green-50' : 'bg-red-50';

  return (
    <Card>
      <h3 className="text-sm font-semibold text-gray-700 mb-4">{displayLabel}</h3>

      <div className="grid grid-cols-2 gap-4 mb-3">
        <div>
          <p className="text-xs text-gray-500 mb-1">Baseline</p>
          <p className="text-2xl font-bold text-gray-900">
            {baselineValue.toFixed(2)}
            {unit && <span className="text-sm font-normal text-gray-500 ml-1">{unit}</span>}
          </p>
        </div>

        <div>
          <p className="text-xs text-gray-500 mb-1">Enhanced</p>
          <p className="text-2xl font-bold text-gray-900">
            {enhancedValue.toFixed(2)}
            {unit && <span className="text-sm font-normal text-gray-500 ml-1">{unit}</span>}
          </p>
        </div>
      </div>

      <div className={`px-3 py-2 rounded-lg ${deltaBgColor}`}>
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-gray-600">Change</span>
          <span className={`text-sm font-bold ${deltaColor}`}>
            {delta > 0 ? '+' : ''}
            {delta.toFixed(1)}%
          </span>
        </div>
      </div>
    </Card>
  );
}
