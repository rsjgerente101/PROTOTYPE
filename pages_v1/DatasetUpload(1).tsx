import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Card } from '../src/components/Card';
import { Button } from '../src/components/Button';
import { Select } from '../src/components/Select';
import { UploadIcon, CheckCircleIcon, AlertCircleIcon, ArrowLeftIcon } from 'lucide-react';
import { MOCK_DATASET } from '../utils/mockData.legacy';

export function DatasetUpload() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [fieldMapping, setFieldMapping] = useState({
    depot_id: 'depot_id',
    depot_lat: 'depot_lat',
    depot_lon: 'depot_lon',
    customer_id: 'customer_id',
    customer_lat: 'customer_lat',
    customer_lon: 'customer_lon',
    order_id: 'order_id',
  });
  const [validation, setValidation] = useState({
    invalid: 0,
    duplicates: 0,
    nearDuplicates: 1,
    isValid: true,
  });
  const [isProcessed, setIsProcessed] = useState(false);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = e.target.files?.[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setTimeout(() => {
        setIsProcessed(true);
        setValidation({
          invalid: 0,
          duplicates: 0,
          nearDuplicates: 1,
          isValid: true,
        });
      }, 1000);
    }
  };

  const fieldOptions = [
    { value: 'depot_id', label: 'depot_id' },
    { value: 'depot_lat', label: 'depot_lat' },
    { value: 'depot_lon', label: 'depot_lon' },
    { value: 'customer_id', label: 'customer_id' },
    { value: 'customer_lat', label: 'customer_lat' },
    { value: 'customer_lon', label: 'customer_lon' },
    { value: 'order_id', label: 'order_id' },
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Load & Map Dataset
          </h1>
          <p className="text-gray-600">
            Upload your reconstructed baseline dataset and configure field mappings
          </p>
        </div>

        {!isProcessed ? (
          // Upload Section
          <Card>
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Upload Dataset
            </h2>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                <UploadIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-sm font-medium text-gray-700 mb-1">
                  Click to upload CSV file
                </p>
                <p className="text-xs text-gray-500">
                  Reconstructed Baseline Dataset (.csv)
                </p>
              </label>
            </div>
          </Card>
        ) : (
          // Processed Content: Single Column Layout
          <div className="space-y-6">
            {/* Dataset Name */}
            {file && (
              <div className="text-gray-700 font-bold text-2xl mb-2">
                Dataset: <span className="font-bold text-2xl">{file.name}</span>
              </div>
            )}

            {/* Field Mapping */}
            <Card>
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Field Mapping
              </h2>
              <div className="space-y-3">
                <Select
                  label="Depot ID"
                  value={fieldMapping.depot_id}
                  onChange={(val) =>
                    setFieldMapping({ ...fieldMapping, depot_id: val })
                  }
                  options={fieldOptions}
                />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <Select
                    label="Depot Latitude"
                    value={fieldMapping.depot_lat}
                    onChange={(val) =>
                      setFieldMapping({ ...fieldMapping, depot_lat: val })
                    }
                    options={fieldOptions}
                  />
                  <Select
                    label="Depot Longitude"
                    value={fieldMapping.depot_lon}
                    onChange={(val) =>
                      setFieldMapping({ ...fieldMapping, depot_lon: val })
                    }
                    options={fieldOptions}
                  />
                </div>
                <Select
                  label="Agent ID"
                  value={fieldMapping.customer_id}
                  onChange={(val) =>
                    setFieldMapping({ ...fieldMapping, customer_id: val })
                  }
                  options={fieldOptions}
                />
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  <Select
                    label="Customer Latitude"
                    value={fieldMapping.customer_lat}
                    onChange={(val) =>
                      setFieldMapping({ ...fieldMapping, customer_lat: val })
                    }
                    options={fieldOptions}
                  />
                  <Select
                    label="Customer Longitude"
                    value={fieldMapping.customer_lon}
                    onChange={(val) =>
                      setFieldMapping({ ...fieldMapping, customer_lon: val })
                    }
                    options={fieldOptions}
                  />
                </div>
                <Select
                  label="Order ID (Optional)"
                  value={fieldMapping.order_id}
                  onChange={(val) =>
                    setFieldMapping({ ...fieldMapping, order_id: val })
                  }
                  options={fieldOptions}
                />
              </div>
            </Card>

            {/* Coordinate Quality Checks */}
            <Card>
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Coordinate Quality Checks
              </h2>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-center gap-2">
                    <CheckCircleIcon className="w-5 h-5 text-green-600" />
                    <span className="text-sm font-medium text-green-900">
                      Invalid Coordinates
                    </span>
                  </div>
                  <span className="text-sm font-semibold text-green-900">
                    {validation.invalid}
                  </span>
                </div>
                <div className="flex items-center justify-between p-3 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-center gap-2">
                    <CheckCircleIcon className="w-5 h-5 text-green-600" />
                    <span className="text-sm font-medium text-green-900">
                      Duplicates
                    </span>
                  </div>
                  <span className="text-sm font-semibold text-green-900">
                    {validation.duplicates}
                  </span>
                </div>
                <div className="flex items-center justify-between p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <div className="flex items-center gap-2">
                    <AlertCircleIcon className="w-5 h-5 text-yellow-600" />
                    <span className="text-sm font-medium text-yellow-900">
                      Near-Duplicates
                    </span>
                  </div>
                  <span className="text-sm font-semibold text-yellow-900">
                    {validation.nearDuplicates}
                  </span>
                </div>
              </div>
            </Card>

            {/* Dataset Summary */}
            <Card>
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Dataset Summary
              </h2>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Total Depots</span>
                  <span className="text-lg font-semibold text-gray-900">
                    {MOCK_DATASET.depots.length}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Total Customers</span>
                  <span className="text-lg font-semibold text-gray-900">
                    {MOCK_DATASET.customers.length}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Total Orders</span>
                  <span className="text-lg font-semibold text-gray-900">
                    {MOCK_DATASET.totalOrders}
                  </span>
                </div>
              </div>
            </Card>

            {/* Proceed Button */}
            <Button
              onClick={() => navigate('/baseline')}
              size="lg"
              className="w-full"
            >
              Proceed to Baseline Run
            </Button>

            {/* Back Button below Proceed */}
            <Button
              onClick={() => setIsProcessed(false)}
              size="lg"
              variant="outline"
              className="w-full flex items-center justify-center gap-2"
            >
              <ArrowLeftIcon className="w-5 h-5" /> Back
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
