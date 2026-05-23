// @ts-nocheck
import React, { useEffect, useMemo, useState } from 'react';
import { Maximize2Icon, Minimize2Icon } from 'lucide-react';
import { MapContainer, TileLayer, Polyline, CircleMarker, Tooltip, useMap } from 'react-leaflet';
import type { Route, Depot, AddedCustomer } from '../types';
import { Card } from './Card';
import MapLegend from './MapLegend';

interface MapCanvasProps {
  routes: Route[];
  depot?: Depot | null;
  showLabels?: boolean;
  showRouteNumbers?: boolean;
  highlightedNodes?: string[];
  onMapReady?: () => void;
  addedCustomers?: AddedCustomer[];
}

const formatSalesRepName = (repId?: string | null) => {
  if (!repId) return '';
  return repId.replace('-AGE-', '-');
};

function toNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function normalizeLatLon(point: any): [number, number] | null {
  if (!point) return null;

  const lat = toNumber(point.lat ?? point.latitude);
  const lon = toNumber(point.lon ?? point.lng ?? point.longitude);

  if (lat == null || lon == null) return null;
  return [lat, lon];
}

function offsetPoint(
  lat: number,
  lon: number,
  routeIndex: number,
  totalRoutes: number
): [number, number] {
  // Small lateral-style visual offset so routes don't perfectly overlap.
  // This is only for map readability, not routing logic.
  const centeredIndex = routeIndex - (totalRoutes - 1) / 2;
  const offsetScale = 0.00035 * centeredIndex;

  return [lat + offsetScale, lon + offsetScale];
}

function buildOffsetPolylinePoints(
  route: Route,
  depot: Depot | null | undefined,
  routeIndex: number,
  totalRoutes: number
): [number, number][] {
  const stopPoints = (route.stops ?? [])
    .map((s) => normalizeLatLon(s))
    .filter((coords): coords is [number, number] => coords !== null)
    .map(([lat, lon]) => offsetPoint(lat, lon, routeIndex, totalRoutes));

  const depotCoords = normalizeLatLon(depot);

  if (depotCoords && stopPoints.length > 0) {
    const depotPt = offsetPoint(depotCoords[0], depotCoords[1], routeIndex, totalRoutes);
    return [depotPt, ...stopPoints, depotPt];
  }

  return stopPoints;
}

function toLatLng(
  path?: Array<{
    lat?: number | string;
    lon?: number | string;
    lng?: number | string;
    latitude?: number | string;
    longitude?: number | string;
  }>
): [number, number][] {
  if (!path || path.length === 0) return [];
  return path
    .map((p) => normalizeLatLon(p))
    .filter((coords): coords is [number, number] => coords !== null);
}

function dedupeAdjacent(points: [number, number][]): [number, number][] {
  if (points.length <= 1) return points;

  const out: [number, number][] = [points[0]];
  for (let i = 1; i < points.length; i += 1) {
    const prev = out[out.length - 1];
    const curr = points[i];
    if (prev[0] !== curr[0] || prev[1] !== curr[1]) {
      out.push(curr);
    }
  }
  return out;
}

function hasOutlierPathPoint(
  points: [number, number][],
  start: [number, number],
  end: [number, number]
): boolean {
  const centerLat = (start[0] + end[0]) / 2;
  const centerLon = (start[1] + end[1]) / 2;

  // City-scale plotting: if a path vertex is far from both endpoints,
  // it's likely an invalid/swapped coordinate and should be ignored.
  return points.some(
    ([lat, lon]) => Math.abs(lat - centerLat) > 1.5 || Math.abs(lon - centerLon) > 1.5
  );
}

function buildAnchoredSegment(
  start: [number, number],
  end: [number, number],
  path?: Array<{
    lat?: number | string;
    lon?: number | string;
    lng?: number | string;
    latitude?: number | string;
    longitude?: number | string;
  }>
): [number, number][] {
  const middle = toLatLng(path);
  if (middle.length === 0) return [start, end];

  if (hasOutlierPathPoint(middle, start, end)) {
    return [start, end];
  }

  return dedupeAdjacent([start, ...middle, end]);
}

function FullscreenMapControl() {
  const map = useMap();
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const updateFullscreenState = () => {
      const mapContainer = map.getContainer();
      setIsFullscreen(document.fullscreenElement === mapContainer);
      window.requestAnimationFrame(() => {
        map.invalidateSize();
      });
    };

    document.addEventListener('fullscreenchange', updateFullscreenState);
    return () => {
      document.removeEventListener('fullscreenchange', updateFullscreenState);
    };
  }, [map]);

  const toggleFullscreen = async () => {
    const mapContainer = map.getContainer();

    try {
      if (document.fullscreenElement === mapContainer) {
        if (document.exitFullscreen) {
          await document.exitFullscreen();
        }
        return;
      }

      if (mapContainer.requestFullscreen) {
        await mapContainer.requestFullscreen();
      }
    } catch (error) {
      console.error('Failed to toggle fullscreen map mode.', error);
      setIsFullscreen(document.fullscreenElement === mapContainer);
      window.requestAnimationFrame(() => {
        map.invalidateSize();
      });
    }
  };

  return (
    <button
      type="button"
      onClick={toggleFullscreen}
      aria-label={isFullscreen ? 'Exit fullscreen map' : 'Open map fullscreen'}
      title={isFullscreen ? 'Exit fullscreen' : 'Fullscreen'}
      className="absolute right-4 top-4 z-[1000] inline-flex items-center justify-center rounded-md border border-gray-200 bg-white/95 p-2 text-gray-700 shadow-sm transition hover:bg-white hover:text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500"
    >
      {isFullscreen ? <Minimize2Icon className="h-4 w-4" /> : <Maximize2Icon className="h-4 w-4" />}
    </button>
  );
}

export function MapCanvas({
  routes,
  depot,
  showLabels = true,
  showRouteNumbers = true,
  highlightedNodes = [],
  onMapReady,
  addedCustomers = [],
}: MapCanvasProps) {
  const allPoints: [number, number][] = useMemo(() => {
    const stopPts = routes
      .flatMap((route) => route.stops ?? [])
      .map((stop) => normalizeLatLon(stop))
      .filter((coords): coords is [number, number] => coords !== null);

    const addedPts = addedCustomers
      .map((c) => normalizeLatLon(c))
      .filter((coords): coords is [number, number] => coords !== null);

    const depotCoords = normalizeLatLon(depot);

    return depotCoords ? [depotCoords, ...stopPts, ...addedPts] : [...stopPts, ...addedPts];
  }, [routes, depot, addedCustomers]);

  const fallbackCenter: [number, number] =
    allPoints.length > 0 ? allPoints[0] : [14.5995, 120.9842];

  const hasHighlightedStop = highlightedNodes.length > 0;

  return (
    <Card className="h-full overflow-hidden" padding="none">
      <div className="relative w-full min-h-[820px] h-[60vh]">
        <MapContainer
          {...(allPoints.length > 0 ? { bounds: allPoints } : { center: fallbackCenter, zoom: 12 })}
          className="w-full h-full rounded-lg"
          scrollWheelZoom
        >
          <TileLayer
            attribution="&copy; OpenStreetMap contributors"
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          <FullscreenMapControl />

          {onMapReady && (
            // child component uses useMap() which must be inside MapContainer
            <MapReadyNotifier onMapReady={onMapReady} />
          )}

          {routes.map((route, routeIndex) => {
            const routeHasHighlightedStop = (route.stops ?? []).some((stop) =>
              highlightedNodes.includes(stop.nodeId)
            );

            const shouldEmphasize = !hasHighlightedStop || routeHasHighlightedStop;

            const fallbackPolylinePoints = buildOffsetPolylinePoints(
              route,
              depot,
              routeIndex,
              routes.length
            );

            const routeStops = route.stops ?? [];
            const depotCoords = normalizeLatLon(depot);

            const stopSegments = routeStops
              .map((stop, idx) => {
                const current = normalizeLatLon(stop);
                if (!current) {
                  return {
                    key: `${route.id}-leg-${idx}`,
                    positions: [] as [number, number][],
                  };
                }

                const prev = idx === 0 ? depotCoords : normalizeLatLon(routeStops[idx - 1]);

                if (!prev) {
                  return {
                    key: `${route.id}-leg-${idx}`,
                    positions: [] as [number, number][],
                  };
                }

                return {
                  key: `${route.id}-leg-${idx}`,
                  positions: buildAnchoredSegment(prev, current, stop.legPath),
                };
              })
              .filter((seg) => seg.positions.length >= 2);

            const lastStopCoords =
              routeStops.length > 0 ? normalizeLatLon(routeStops[routeStops.length - 1]) : null;

            const returnSegment =
              lastStopCoords && depotCoords
                ? buildAnchoredSegment(lastStopCoords, depotCoords, route.returnPath)
                : [];

            if (stopSegments.length === 0) {
              return (
                <Polyline
                  key={route.id}
                  positions={fallbackPolylinePoints}
                  pathOptions={{
                    color: route.color,
                    weight: shouldEmphasize ? 5 : 3,
                    opacity: shouldEmphasize ? 0.9 : 0.35,
                  }}
                />
              );
            }

            return (
              <React.Fragment key={route.id}>
                {stopSegments.map((seg) => (
                  <Polyline
                    key={seg.key}
                    positions={seg.positions}
                    pathOptions={{
                      color: route.color,
                      weight: shouldEmphasize ? 5 : 3,
                      opacity: shouldEmphasize ? 0.9 : 0.35,
                    }}
                  />
                ))}

                {returnSegment.length >= 2 && (
                  <Polyline
                    key={`${route.id}-return`}
                    positions={returnSegment}
                    pathOptions={{
                      color: route.color,
                      weight: shouldEmphasize ? 5 : 3,
                      opacity: shouldEmphasize ? 0.9 : 0.35,
                    }}
                  />
                )}
              </React.Fragment>
            );
          })}

          {normalizeLatLon(depot) && (
            <CircleMarker
              center={normalizeLatLon(depot) as [number, number]}
              radius={8}
              pathOptions={{
                color: '#fff',
                fillColor: '#DC2626',
                fillOpacity: 1,
                weight: 2,
              }}
            >
              {showLabels && (
                <Tooltip permanent direction="top">
                  Depot
                </Tooltip>
              )}
            </CircleMarker>
          )}

          {routes.map((route) =>
            (route.stops ?? []).map((stop) => {
              const coords = normalizeLatLon(stop);
              if (!coords) return null;

              const isHighlighted = highlightedNodes.includes(stop.nodeId);

              return (
                <CircleMarker
                  key={`${route.id}-${stop.nodeId}-${stop.stopNumber}`}
                  center={coords}
                  radius={isHighlighted ? 8 : 5}
                  pathOptions={{
                    color: isHighlighted ? '#111827' : '#fff',
                    fillColor: isHighlighted ? '#F59E0B' : route.color,
                    fillOpacity: 1,
                    weight: isHighlighted ? 3 : 1,
                  }}
                >
                  {showLabels && (
                    <Tooltip direction="top" offset={[0, -5]} permanent>
                      {`${stop.nodeName} - ${formatSalesRepName(route.representativeId)}`}
                    </Tooltip>
                  )}

                  {showRouteNumbers && (
                    <Tooltip direction="center" permanent>
                      {stop.stopNumber}
                    </Tooltip>
                  )}
                </CircleMarker>
              );
            })
          )}

          {addedCustomers.map((customer) => {
            const coords = normalizeLatLon(customer);
            if (!coords) return null;

            return (
              <CircleMarker
                key={customer.id}
                center={coords}
                radius={6}
                pathOptions={{
                  color: '#111827',
                  fillColor: '#10B981',
                  fillOpacity: 1,
                  weight: 2,
                }}
              >
                {showLabels && (
                  <Tooltip direction="top" offset={[0, -5]} permanent>
                    {customer.label}
                  </Tooltip>
                )}
              </CircleMarker>
            );
          })}
        </MapContainer>

        <div className="absolute bottom-55 right-4 z-[999]">
          <MapLegend />
        </div>
      </div>
    </Card>
  );
}

function MapReadyNotifier({ onMapReady }: { onMapReady?: () => void }) {
  const map = useMap();

  useEffect(() => {
    if (!onMapReady || !map) return;

    try {
      map.whenReady(() => {
        onMapReady();
      });
    } catch (err) {
      // fallback: call onMapReady once if whenReady isn't available
      onMapReady();
    }
  }, [map, onMapReady]);

  return null;
}
