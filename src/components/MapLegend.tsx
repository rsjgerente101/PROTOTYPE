import React from 'react';

const MapLegend = () => {
  return (
    <div className="map-legend">
      <h4 className="font-semibold mb-2">Map Legend</h4>

      <div className="legend-item">
        <span className="dot depot"></span> Depot
      </div>

      <div className="legend-item">
        <span className="dot stop"></span> Customer Stop
      </div>

      <div className="legend-item">
        <span className="dot highlighted"></span> Highlighted / Selected Stop
      </div>

      <div className="legend-item">
        <span className="line route"></span> Route Path
      </div>
    </div>
  );
};

export default MapLegend;
