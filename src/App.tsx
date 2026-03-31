import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import DatasetUpload from './pages/DatasetUpload';
import BaselineRun from './pages/BaselineRun';
import EnhancedRun from './pages/EnhancedRun';
import CompareExport from './pages/CompareExport';

export default function App() { 
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<DatasetUpload />} />
        <Route path="/baseline" element={<BaselineRun />} />
        <Route path="/enhanced" element={<EnhancedRun />} />
        <Route path="/compare" element={<CompareExport />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}