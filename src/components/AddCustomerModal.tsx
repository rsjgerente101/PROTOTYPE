import React, { useEffect, useMemo, useState } from 'react';
import { XIcon } from 'lucide-react';
import { MapContainer, TileLayer, Marker, useMapEvents, useMap } from 'react-leaflet';
import L from 'leaflet';
import type { Depot } from '../types';

// Use an inline SVG DivIcon so we don't depend on image files
const pinSvg = `
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="28" height="40" aria-hidden="true">
    <path fill="%23ef4444" d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/>
    <circle fill="%23ffffff" cx="12" cy="9" r="2.5"/>
  </svg>
`;

const DefaultIcon = L.divIcon({
  className: 'custom-leaflet-pin',
  html: pinSvg,
  iconSize: [28, 40],
  iconAnchor: [14, 40],
});

L.Marker.prototype.options.icon = DefaultIcon;

type PickLocation = {
  lat?: number | null;
  lon?: number | null;
  address?: string;
  locationSet?: boolean;
};

export type AddCustomer = {
  label: string;
  lat?: number | null;
  lon?: number | null;
  address?: string;
};

interface AddCustomerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (customers: AddCustomer[]) => void;
  depot?: Depot | null;
}

function useNominatimSearch() {
  const [results, setResults] = useState<any[]>([]);
  const search = async (q: string) => {
    if (!q || q.trim().length === 0) {
      setResults([]);
      return;
    }
    try {
      const url = `https://nominatim.openstreetmap.org/search?format=json&addressdetails=1&limit=5&q=${encodeURIComponent(
        q
      )}`;
      const res = await fetch(url, {
        headers: {
          'Accept-Language': 'en',
        },
      });
      const json = await res.json();
      setResults(json || []);
    } catch (e) {
      setResults([]);
    }
  };
  return { results, search, setResults };
}

function SimplifiedPicker({
  value,
  onChange,
  label,
  depot,
}: {
  value: PickLocation;
  onChange: (v: PickLocation) => void;
  label: string;
  depot?: Depot | null;
}) {
  const { results, search, setResults } = useNominatimSearch();
  const center: [number, number] = useMemo(() => {
    if (value.lat != null && value.lon != null) return [value.lat, value.lon];
    if (depot && depot.lat != null && depot.lon != null) return [depot.lat, depot.lon];
    return [14.5995, 120.9842];
  }, [value, depot]);

  function MapEvents() {
    useMapEvents({
      click(e) {
        const lat = e.latlng.lat;
        const lon = e.latlng.lng;
        onChange({ lat, lon, address: undefined, locationSet: true });
        setResults([]);
      },
    });
    return null;
  }

  // new: initialize map and call invalidateSize via useMap()
  function MapInitializer({ val }: { val: PickLocation }) {
    const map = useMap();
    useEffect(() => {
      if (val.lat != null && val.lon != null) {
        map.setView([val.lat, val.lon], 14);
      }
      const t = setTimeout(() => {
        try {
          map.invalidateSize();
        } catch { }
      }, 50);
      return () => clearTimeout(t);
    }, [map, val.lat, val.lon]);
    return null;
  }

  return (
    <div className="rounded-md p-3">
      <div className="mb-2 relative">
        <input
          placeholder="Search place or address"
          className="w-full rounded border border-slate-200 px-2 py-1 text-sm"
          onChange={(e) => search(e.target.value)}
        />
        {results.length > 0 && (
          <div className="absolute left-0 right-0 mt-1 bg-white border border-slate-200 rounded max-h-40 overflow-auto z-[9999]">
            {results.map((r: any) => (
              <button
                key={r.place_id}
                type="button"
                className="block w-full text-left px-2 py-1 hover:bg-slate-100 text-sm"
                onClick={() => {
                  const lat = Number(r.lat);
                  const lon = Number(r.lon);
                  const display = r.display_name;
                  onChange({ lat, lon, address: display, locationSet: true });
                  setResults([]);
                }}
              >
                {r.display_name}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="mb-2 rounded overflow-hidden" style={{ height: '40vh', minHeight: 200 }}>
        <MapContainer center={center} zoom={13} style={{ height: '100%', width: '100%' }}>
          <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
          <MapEvents />
          <MapInitializer val={value} />
          {depot && depot.lat != null && depot.lon != null && (
            <Marker position={[depot.lat, depot.lon]} />
          )}
          {value.lat != null && value.lon != null && (
            <Marker
              draggable
              position={[Number(value.lat), Number(value.lon)]}
              eventHandlers={{
                dragend: (e) => {
                  const m = e.target as L.Marker;
                  const p = m.getLatLng();
                  onChange({
                    lat: p.lat,
                    lon: p.lng,
                    address: value.address,
                    locationSet: true,
                  });
                },
              }}
            />
          )}
        </MapContainer>
      </div>

      <div className="text-xs text-gray-600 mb-1">
        Search a place or click on the map to pin the customer location
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <div className="text-xs text-gray-500">Latitude</div>
          <div>{value.lat != null ? value.lat.toFixed(6) : '-'}</div>
        </div>
        <div>
          <div className="text-xs text-gray-500">Longitude</div>
          <div>{value.lon != null ? value.lon.toFixed(6) : '-'}</div>
        </div>
      </div>
    </div>
  );
}

export default function AddCustomerModal({
  isOpen,
  onClose,
  onConfirm,
  depot,
}: AddCustomerModalProps) {
  const [count, setCount] = useState(1);
  const [pickers, setPickers] = useState<PickLocation[]>([
    { locationSet: false },
  ] as PickLocation[]);

  useEffect(() => {
    setPickers(
      Array.from({ length: Math.max(1, count) }, (_, i) => pickers[i] ?? { locationSet: false })
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [count]);

  useEffect(() => {
    if (!isOpen) {
      setCount(1);
      setPickers([{ locationSet: false }]);
    }
  }, [isOpen]);

  // When modal opens, give anchors a chance to render and then invalidate any inner Leaflet maps.
  useEffect(() => {
    if (!isOpen) return;
    // small delay to allow MapContainer whenCreated callbacks to run, they already call invalidateSize.
    const t = setTimeout(() => {
      // attempt to trigger global resize for Leaflet maps
      try {
        window.dispatchEvent(new Event('resize'));
      } catch (e) {
        // ignore
      }
    }, 120);
    return () => clearTimeout(t);
  }, [isOpen]);

  function updatePicker(idx: number, v: PickLocation) {
    setPickers((prev) => {
      const copy = prev.slice();
      copy[idx] = { ...copy[idx], ...v };
      return copy;
    });
  }

  const allValid =
    pickers.length > 0 && pickers.every((p) => p.lat != null && p.lon != null && p.locationSet);

  return !isOpen ? null : (
    <div className="fixed inset-0 z-[1500] flex items-start justify-center p-6">
      <div className="absolute inset-0 bg-transparent" onClick={onClose} />

      <div className="fixed inset-0 z-[1500] flex items-start justify-center p-6">
        <div className="absolute inset-0 bg-transparent" onClick={onClose} />

        <div className="relative w-full max-w-4xl bg-white rounded-lg ring-1 ring-gray-200 z-[1501] flex flex-col max-h-[90vh]">
          <div className="sticky top-0 bg-white z-30 border-slate-200 px-4 py-3">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold">Add Customer</h3>
              <button onClick={onClose} className="text-gray-600 hover:text-gray-900">
                <XIcon />
              </button>
            </div>

            <div className="mt-3">
              <label className="block text-sm text-gray-700 mb-1">
                How many customers do you want to add?
              </label>
              <input
                type="number"
                min={1}
                value={count}
                className="w-24 rounded border border-slate-200 px-2 py-1"
                onChange={(e) => setCount(Math.max(1, Number(e.target.value) || 1))}
              />
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4">
            {Array.from({ length: count }).map((_, idx) => (
              <div
                key={idx}
                className="border border-slate-200 rounded p-3 overflow-hidden relative z-0"
              >
                <div className="mb-2 font-medium">Customer {idx + 1}</div>

                <SimplifiedPicker
                  label={`Customer ${idx + 1}`}
                  value={pickers[idx] ?? { locationSet: false }}
                  onChange={(v) => updatePicker(idx, v)}
                  depot={depot}
                />
              </div>
            ))}
          </div>

          <div className="sticky bottom-0 bg-white border-slate-200 px-4 py-3 flex justify-end gap-3 z-30">
            <button
              type="button"
              className="px-4 py-2 rounded bg-white border border-slate-200 text-sm"
              onClick={onClose}
            >
              Cancel
            </button>
            <button
              type="button"
              className={`px-4 py-2 rounded text-white text-sm ${allValid ? 'bg-blue-600' : 'bg-slate-300 cursor-not-allowed'
                }`}
              onClick={() => {
                if (!allValid) return;
                const customers: AddCustomer[] = pickers.map((p, i) => ({
                  label: `Customer ${i + 1}`,
                  lat: p.lat ?? null,
                  lon: p.lon ?? null,
                  address: p.address,
                }));
                onClose();
                Promise.resolve(onConfirm(customers)).catch(() => {
                  // swallow errors to avoid unhandled rejections; parent handles errors
                });
              }}
              disabled={!allValid}
            >
              Confirm Add Customers
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
