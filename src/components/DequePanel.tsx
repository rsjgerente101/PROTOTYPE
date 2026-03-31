import React from 'react';
import type { Representative } from '../types';
import { RepresentativeCard } from './RepresentativeCard';
import { Card } from './Card';
import { Button } from './Button';
import { ArrowLeftIcon, ArrowRightIcon } from 'lucide-react';

interface DequePanelProps {
  representatives: Representative[];
  onUpdate?: (reps: Representative[]) => void;
}

export function DequePanel({ representatives, onUpdate }: DequePanelProps) {
  const handlePushFront = (repId: string) => {
    if (!onUpdate) return;

    const updatedReps = [...representatives];
    const repIndex = updatedReps.findIndex((r) => r.id === repId);

    if (repIndex > 0) {
      const [rep] = updatedReps.splice(repIndex, 1);
      updatedReps.unshift(rep);
      onUpdate(updatedReps);
    }
  };

  const handlePushBack = (repId: string) => {
    if (!onUpdate) return;

    const updatedReps = [...representatives];
    const repIndex = updatedReps.findIndex((r) => r.id === repId);

    if (repIndex < updatedReps.length - 1) {
      const [rep] = updatedReps.splice(repIndex, 1);
      updatedReps.push(rep);
      onUpdate(updatedReps);
    }
  };

  return (
    <Card title="Deque Panel">
      <div className="mb-4">
        <p className="text-xs text-gray-600">
          Manage workload distribution across representatives
        </p>
      </div>

      <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-center justify-between text-xs">
          <span className="font-medium text-blue-900">Front of Queue</span>
          <ArrowRightIcon className="w-4 h-4 text-blue-600" />
          <span className="font-medium text-blue-900">Rear of Queue</span>
        </div>
      </div>

      <div className="space-y-3 max-h-96 overflow-y-auto">
        {representatives.map((rep, index) => (
          <div key={rep.id} className="relative">
            <div className="absolute -left-8 top-1/2 -translate-y-1/2 text-xs font-semibold text-gray-400">
              {index === 0
                ? 'F'
                : index === representatives.length - 1
                ? 'R'
                : index + 1}
            </div>

            <RepresentativeCard representative={rep} isDraggable={!!onUpdate} />

            <div className="flex gap-2 mt-2">
              <Button
                size="sm"
                variant="outline"
                onClick={() => handlePushFront(rep.id)}
                disabled={!onUpdate || index === 0}
                className="flex-1 flex items-center justify-center text-xs gap-1"
              >
                <ArrowLeftIcon className="w-3 h-3" />
                Push Front
              </Button>

              <Button
                size="sm"
                variant="outline"
                onClick={() => handlePushBack(rep.id)}
                disabled={!onUpdate || index === representatives.length - 1}
                className="flex-1 flex items-center justify-center text-xs gap-1"
              >
                Push Back
                <ArrowRightIcon className="w-3 h-3" />
              </Button>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}