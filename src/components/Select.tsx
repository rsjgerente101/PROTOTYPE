import React from 'react';
interface SelectProps {
  label?: string;
  value: string;
  onChange: (value: string) => void;
  options: {
    value: string;
    label: string;
  }[];
  disabled?: boolean;
  className?: string;
}
export function Select({
  label,
  value,
  onChange,
  options,
  disabled = false,
  className = ''
}: SelectProps) {
  return <div className={`flex flex-col gap-1.5 ${className}`}>
      {label && <label className="text-sm font-medium text-gray-700">{label}</label>}
      <select value={value} onChange={e => onChange(e.target.value)} disabled={disabled} className={`px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent ${disabled ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}`}>
        {options.map(option => <option key={option.value} value={option.value}>
            {option.label}
          </option>)}
      </select>
    </div>;
}