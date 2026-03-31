import React from 'react';
import { TrendingUpIcon, TrendingDownIcon } from 'lucide-react';
import { Card } from './Card';
interface KPICardProps {
  title: string;
  value: string | number;
  unit?: string;
  delta?: number;
  deltaLabel?: string;
  icon?: React.ReactNode;
}
export function KPICard({
  title,
  value,
  unit,
  delta,
  deltaLabel,
  icon
}: KPICardProps) {
  const showDelta = delta !== undefined;
  const isPositive = delta && delta > 0;
  const isNegative = delta && delta < 0;
  return <Card>
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600">{title}</p>
          <div className="mt-2 flex items-baseline gap-2">
            <p className="text-2xl font-semibold text-gray-900">{value}</p>
            {unit && <span className="text-sm text-gray-500">{unit}</span>}
          </div>
          {showDelta && <div className="mt-2 flex items-center gap-1">
              {isPositive && <TrendingUpIcon className="w-4 h-4 text-green-600" />}
              {isNegative && <TrendingDownIcon className="w-4 h-4 text-red-600" />}
              <span className={`text-sm font-medium ${isPositive ? 'text-green-600' : isNegative ? 'text-red-600' : 'text-gray-600'}`}>
                {delta > 0 ? '+' : ''}
                {delta}%
              </span>
              {deltaLabel && <span className="text-sm text-gray-500">{deltaLabel}</span>}
            </div>}
        </div>
        {icon && <div className="text-gray-400">{icon}</div>}
      </div>
    </Card>;
}