import React from 'react';
interface InputProps {
  label?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  type?: 'text' | 'number' | 'email';
  disabled?: boolean;
  error?: string;
  className?: string;
}
export function Input({
  label,
  value,
  onChange,
  placeholder,
  type = 'text',
  disabled = false,
  error,
  className = ''
}: InputProps) {
  return <div className={`flex flex-col gap-1.5 ${className}`}>
      {label && <label className="text-sm font-medium text-gray-700">{label}</label>}
      <input type={type} value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} disabled={disabled} className={`px-3 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent ${error ? 'border-red-500' : 'border-gray-300'} ${disabled ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}`} />
      {error && <span className="text-sm text-red-600">{error}</span>}
    </div>;
}