import React from 'react';
import { Representative } from '../types';
import { UserIcon } from 'lucide-react';

interface RepresentativeCardProps {
  representative: Representative;
  onReassign?: (repId: string) => void;
  isDraggable?: boolean;
}

export function RepresentativeCard({
  representative,
  onReassign,
  isDraggable = false,
}: RepresentativeCardProps) {
  const color = representative.color ?? '#2563eb';

  return (
    <div
      className={`bg-white border-2 rounded-lg p-3 ${isDraggable ? 'cursor-move' : ''}`}
      style={{ borderColor: color }}
      draggable={isDraggable}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <div
            className="w-8 h-8 rounded-full flex items-center justify-center"
            style={{ backgroundColor: `${color}20` }}
          >
            <UserIcon className="w-4 h-4" style={{ color }} />
          </div>
          <div>
            <p className="text-sm font-semibold text-gray-900">
              {representative.name}
            </p>
            <p className="text-xs text-gray-500">{representative.id}</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 mt-3">
        <div>
          <p className="text-xs text-gray-500">Workload</p>
          <p className="text-sm font-semibold text-gray-900">
            {representative.workload.toFixed(1)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Opportunity</p>
          <p className="text-sm font-semibold text-gray-900">
            {representative.opportunityScore.toFixed(1)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Priority</p>
          <p className="text-sm font-semibold text-gray-900">
            {representative.priorityScore.toFixed(1)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Queue Pos.</p>
          <p className="text-sm font-semibold text-gray-900">
            {representative.queuePosition}
          </p>
        </div>
      </div>

      <div className="mt-3 pt-3 border-t border-gray-200">
        <p className="text-xs text-gray-500 mb-1">Assigned Customers</p>
        <p className="text-sm font-semibold text-gray-900">
          {representative.assignedCustomers}
        </p>
      </div>
    </div>
  );
}