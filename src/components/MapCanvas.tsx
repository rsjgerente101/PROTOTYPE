import React, { useMemo } from 'react';
import { MapContainer, TileLayer, Polyline, CircleMarker, Tooltip } from 'react-leaflet';
import { Route, Depot } from '../types';
import { Card } from './Card';
import MapLegend from './MapLegend';

interface MapCanvasProps {
  routes: Route[];
  depot?: Depot | null;
  showLabels?: boolean;
  showRouteNumbers?: boolean;
  highlightedNodes?: string[];
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
    .filter((s) => typeof s.lat === 'number' && typeof s.lon === 'number')
    .map((s) => offsetPoint(s.lat, s.lon, routeIndex, totalRoutes));

  if (depot && stopPoints.length > 0) {
    const depotPt = offsetPoint(depot.lat, depot.lon, routeIndex, totalRoutes);
    return [depotPt, ...stopPoints, depotPt];
  }

  return stopPoints;
}

export function MapCanvas({
  routes,
  depot,
  showLabels = true,
  showRouteNumbers = true,
  highlightedNodes = [],
}: MapCanvasProps) {
  const allStops = routes.flatMap((r) => r.stops ?? []);

  const allPoints: [number, number][] = useMemo(() => {
    const stopPts = allStops
      .filter((s) => typeof s.lat === 'number' && typeof s.lon === 'number')
      .map((s) => [s.lat, s.lon] as [number, number]);

    return depot ? [[depot.lat, depot.lon], ...stopPts] : stopPts;
  }, [allStops, depot]);

  const fallbackCenter: [number, number] =
    allPoints.length > 0 ? allPoints[0] : [14.5995, 120.9842];

  const hasHighlightedStop = highlightedNodes.length > 0;

  return (
    <Card className="bg-white rounded-lg shadow-sm border border-gray-200 h-full" padding="none">
      <div className="relative w-full h-full">
        <MapContainer
          {...(allPoints.length > 0
            ? { bounds: allPoints }
            : { center: fallbackCenter, zoom: 12 })}
          className="w-full h-full rounded-lg"
          scrollWheelZoom
        >
          <TileLayer
            attribution="&copy; OpenStreetMap contributors"
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {routes.map((route, routeIndex) => {
            const polylinePoints = buildOffsetPolylinePoints(
              route,
              depot,
              routeIndex,
              routes.length
            );

            const routeHasHighlightedStop =
              (route.stops ?? []).some((stop) => highlightedNodes.includes(stop.nodeId));

            const shouldEmphasize = !hasHighlightedStop || routeHasHighlightedStop;

            return (
              <Polyline
                key={route.id}
                positions={polylinePoints}
                pathOptions={{
                  color: route.color,
                  weight: shouldEmphasize ? 5 : 3,
                  opacity: shouldEmphasize ? 0.9 : 0.35,
                }}
              />
            );
          })}

          {depot && (
            <CircleMarker
              center={[depot.lat, depot.lon]}
              radius={8}
              pathOptions={{
                color: '#fff',
                fillColor: '#DC2626',
                fillOpacity: 1,
                weight: 2,
              }}
            >
              {showLabels && <Tooltip permanent direction="top">Depot</Tooltip>}
            </CircleMarker>
          )}

          {routes.map((route) =>
            (route.stops ?? []).map((stop) => {
              const isHighlighted = highlightedNodes.includes(stop.nodeId);

              return (
                <CircleMarker
                  key={`${route.id}-${stop.nodeId}-${stop.stopNumber}`}
                  center={[stop.lat, stop.lon]}
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
                      {stop.nodeName}
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
        </MapContainer>

        <div className="absolute bottom-4 right-4 z-[999]">
          <MapLegend />
        </div>
      </div>
    </Card>
  );
}