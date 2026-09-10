#!/usr/bin/env python3
"""
wienmobil_router.py
====================

Plans a cost-optimized bicycle route in Vienna using the WienMobil Rad
(nextbike) bike-share system, ensuring no single ride leg exceeds a
configurable time limit (default: 25 minutes, to stay safely under
nextbike's 30-minute free-ride window).

How it works
------------
1. Request a cycling route from Google Maps Directions API (mode=bicycling).
2. If the route's total duration is within the limit, it's returned as-is.
3. Otherwise, the route's overview polyline is decoded into a Shapely
   LineString. We interpolate the point along that line corresponding to
   the fraction (search_minutes / total_duration_minutes) -- i.e. roughly
   where you'd be after `search_minutes` of riding, ASSUMING a fairly
   constant speed (this is an approximation: interpolation is done by
   distance fraction along the polyline, which is a good proxy for time
   fraction on a single continuous cycling leg with no major stops).
4. The live WienMobil Rad / nextbike station feed is queried, and the
   nearest station(s) with available bikes to that interpolated point are
   found using the Haversine formula.
5. A handful of nearby candidate stations are evaluated by actually
   requesting bike directions to each of them, and the best one (longest
   ride that still stays within the time limit) is chosen as a mandatory
   transfer point.
6. The process repeats from that station to the final destination until
   the remaining leg is under the time limit, or a safety cap on the
   number of legs is hit.

Requirements
------------
See requirements.txt. You need your own Google Maps API key with the
Directions API enabled and billing configured.

Usage
-----
    export GOOGLE_MAPS_API_KEY="your-key-here"
    python3 wienmobil_router.py "Stephansplatz, Wien" "Schoenbrunn, Wien"

or pass the key explicitly:

    python3 wienmobil_router.py "Start address" "Destination address" \\
        --api-key YOUR_KEY

See --help for all options.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlencode

import requests

try:
    import polyline as polyline_lib
except ImportError:  # pragma: no cover
    print("Missing dependency 'polyline'. Install with: pip install polyline", file=sys.stderr)
    raise

try:
    from shapely.geometry import LineString, Point
except ImportError:  # pragma: no cover
    print("Missing dependency 'shapely'. Install with: pip install shapely", file=sys.stderr)
    raise


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

GOOGLE_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
NEXTBIKE_LIVE_URL = "https://maps.nextbike.net/maps/nextbike-live.json?city=748"
EARTH_RADIUS_KM = 6371.0088
REQUEST_TIMEOUT_S = 15
USER_AGENT = "WienMobilRouter/1.0 (+personal-use-script)"

DEFAULT_MAX_RIDE_MINUTES = 25.0       # hard cap per leg
DEFAULT_SEARCH_MINUTES = 22.0         # where along the route we start looking for a station
DEFAULT_MAX_LEGS = 8                  # safety cap against infinite loops
DEFAULT_SEARCH_RADIUS_KM = 2.0        # how far we're willing to detour to a station
DEFAULT_CANDIDATE_STATIONS = 5        # how many nearby stations to actually test


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class RouteNotFoundError(Exception):
    """Raised when Google Directions can't produce a bicycling route."""


class NextbikeAPIError(Exception):
    """Raised when the nextbike/WienMobil feed can't be read or is empty."""


class NoStationFoundError(Exception):
    """Raised when no suitable transfer station can be found near a point."""


# --------------------------------------------------------------------------
# Google Directions
# --------------------------------------------------------------------------

def get_directions(session: requests.Session, origin: str, destination: str, api_key: str) -> Dict:
    """Fetch a single bicycling route leg from Google Directions API.

    `origin` and `destination` can be free-text addresses or "lat,lng" strings.
    """
    params = {
        "origin": origin,
        "destination": destination,
        "mode": "bicycling",
        "key": api_key,
    }
    try:
        resp = session.get(GOOGLE_DIRECTIONS_URL, params=params, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise RouteNotFoundError(f"Network error requesting directions {origin!r} -> {destination!r}: {exc}") from exc
    except ValueError as exc:
        raise RouteNotFoundError(f"Could not parse Directions API response for {origin!r} -> {destination!r}: {exc}") from exc

    status = data.get("status")
    if status != "OK":
        error_message = data.get("error_message", "")
        raise RouteNotFoundError(
            f"Google Directions API returned '{status}' for {origin!r} -> {destination!r}. {error_message}".strip()
        )

    routes = data.get("routes") or []
    if not routes:
        raise RouteNotFoundError(f"No routes returned for {origin!r} -> {destination!r}.")

    route = routes[0]
    legs = route.get("legs") or []
    if not legs:
        raise RouteNotFoundError(f"Route had no legs for {origin!r} -> {destination!r}.")

    leg = legs[0]
    overview_polyline = route.get("overview_polyline", {}).get("points")
    if not overview_polyline:
        raise RouteNotFoundError(f"Route had no overview polyline for {origin!r} -> {destination!r}.")

    return {
        "duration_sec": leg["duration"]["value"],
        "distance_m": leg["distance"]["value"],
        "start_address": leg.get("start_address", origin),
        "end_address": leg.get("end_address", destination),
        "overview_polyline": overview_polyline,
    }


# --------------------------------------------------------------------------
# Polyline / geometry helpers
# --------------------------------------------------------------------------

def decode_polyline(encoded: str) -> List[Tuple[float, float]]:
    """Decode a Google encoded polyline into a list of (lat, lon) tuples."""
    return polyline_lib.decode(encoded)


def build_linestring(coords_lat_lon: Sequence[Tuple[float, float]]) -> LineString:
    """Build a Shapely LineString (x=lon, y=lat) from (lat, lon) coordinates."""
    return LineString([(lon, lat) for lat, lon in coords_lat_lon])


def interpolate_point(line: LineString, fraction: float) -> Tuple[float, float]:
    """Interpolate a point at `fraction` (0-1) of the line's length.

    Returns (lat, lon).
    """
    fraction = min(max(fraction, 0.0), 1.0)
    point: Point = line.interpolate(fraction, normalized=True)
    return point.y, point.x  # (lat, lon)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------
# WienMobil Rad / nextbike live data
# --------------------------------------------------------------------------

def fetch_nextbike_stations(session: requests.Session) -> List[Dict]:
    """Fetch all WienMobil Rad places with at least one available bike."""
    try:
        resp = session.get(
            NEXTBIKE_LIVE_URL,
            timeout=REQUEST_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as exc:
        raise NextbikeAPIError(f"Network error fetching nextbike live data: {exc}") from exc
    except ValueError as exc:
        raise NextbikeAPIError(f"Could not parse nextbike live data: {exc}") from exc

    stations: List[Dict] = []
    for country in data.get("countries", []):
        for city in country.get("cities", []):
            for place in city.get("places", []):
                bikes = place.get("bikes", 0) or 0
                lat, lng = place.get("lat"), place.get("lng")
                if bikes <= 0 or lat is None or lng is None or place.get("bike"):
                    continue
                stations.append(
                    {
                        "uid": place.get("uid"),
                        "name": place.get("name") or "Unnamed WienMobil station",
                        "lat": float(lat),
                        "lng": float(lng),
                        "bikes": int(bikes),
                    }
                )

    if not stations:
        raise NextbikeAPIError("No active WienMobil Rad stations with available bikes were found.")

    return stations


def find_candidate_stations(
    point_lat: float,
    point_lon: float,
    stations: Sequence[Dict],
    exclude_uids: Set,
    top_n: int = DEFAULT_CANDIDATE_STATIONS,
) -> List[Tuple[float, Dict]]:
    """Return up to `top_n` (distance_km, station) pairs nearest to the given point."""
    scored = []
    for station in stations:
        if station["uid"] in exclude_uids:
            continue
        dist_km = haversine_km(point_lat, point_lon, station["lat"], station["lng"])
        scored.append((dist_km, station))
    scored.sort(key=lambda pair: pair[0])
    return scored[:top_n]


# --------------------------------------------------------------------------
# Google Maps link building
# --------------------------------------------------------------------------

def build_maps_url(origin_query: str, destination_query: str) -> str:
    """Build a clickable Google Maps directions URL for a bicycling leg."""
    params = {
        "api": "1",
        "origin": origin_query,
        "destination": destination_query,
        "travelmode": "bicycling",
    }
    return "https://www.google.com/maps/dir/?" + urlencode(params)


# --------------------------------------------------------------------------
# Route planning
# --------------------------------------------------------------------------

def build_leg_record(
    leg_num: int,
    origin_label: str,
    destination_label: str,
    origin_query: str,
    destination_query: str,
    directions: Dict,
    station_swap: Optional[str],
    station_bikes: Optional[int] = None,
) -> Dict:
    return {
        "leg": leg_num,
        "origin_label": origin_label,
        "destination_label": destination_label,
        "duration_min": directions["duration_sec"] / 60.0,
        "distance_km": directions["distance_m"] / 1000.0,
        "maps_url": build_maps_url(origin_query, destination_query),
        "station_swap": station_swap,
        "station_bikes": station_bikes,
    }


def plan_route(
    start: str,
    destination: str,
    api_key: str,
    max_ride_minutes: float = DEFAULT_MAX_RIDE_MINUTES,
    search_minutes: float = DEFAULT_SEARCH_MINUTES,
    max_legs: int = DEFAULT_MAX_LEGS,
    search_radius_km: float = DEFAULT_SEARCH_RADIUS_KM,
) -> List[Dict]:
    """Plan a full trip made of one or more <= max_ride_minutes legs."""
    session = requests.Session()
    stations = fetch_nextbike_stations(session)

    used_uids: Set = set()
    legs: List[Dict] = []

    current_origin_query = start
    current_origin_label = start

    for leg_num in range(1, max_legs + 1):
        directions = get_directions(session, current_origin_query, destination, api_key)
        duration_min = directions["duration_sec"] / 60.0

        if duration_min <= max_ride_minutes:
            legs.append(
                build_leg_record(
                    leg_num,
                    current_origin_label,
                    directions["end_address"],
                    current_origin_query,
                    destination,
                    directions,
                    station_swap=None,
                )
            )
            return legs

        # Route is too long for a single leg -- find a transfer station.
        fraction = search_minutes / duration_min
        coords = decode_polyline(directions["overview_polyline"])
        line = build_linestring(coords)
        point_lat, point_lon = interpolate_point(line, fraction)

        candidates = find_candidate_stations(point_lat, point_lon, stations, used_uids)
        candidates = [(d, s) for d, s in candidates if d <= search_radius_km]
        if not candidates:
            raise NoStationFoundError(
                f"No WienMobil Rad station with available bikes found within "
                f"{search_radius_km:.1f} km of the ~{search_minutes:.0f}-minute point on leg {leg_num}."
            )

        evaluated = []
        for dist_km, station in candidates:
            station_query = f"{station['lat']},{station['lng']}"
            try:
                station_directions = get_directions(session, current_origin_query, station_query, api_key)
            except RouteNotFoundError:
                continue
            evaluated.append((station_directions["duration_sec"] / 60.0, dist_km, station, station_directions))

        if not evaluated:
            raise NoStationFoundError(
                f"Could not compute a valid bike route to any nearby station on leg {leg_num}."
            )

        within_limit = [e for e in evaluated if e[0] <= max_ride_minutes]
        if within_limit:
            # Prefer the candidate that covers the most ground while staying under the limit.
            within_limit.sort(key=lambda e: -e[0])
            chosen_duration, chosen_dist, chosen_station, chosen_directions = within_limit[0]
        else:
            # Nothing fits cleanly -- fall back to the fastest option and warn.
            evaluated.sort(key=lambda e: e[0])
            chosen_duration, chosen_dist, chosen_station, chosen_directions = evaluated[0]
            print(
                f"  Warning: nearest reachable station for leg {leg_num} is "
                f"{chosen_duration:.1f} min away, slightly over the {max_ride_minutes:.0f}-min target.",
                file=sys.stderr,
            )

        station_query = f"{chosen_station['lat']},{chosen_station['lng']}"
        legs.append(
            build_leg_record(
                leg_num,
                current_origin_label,
                chosen_station["name"],
                current_origin_query,
                station_query,
                chosen_directions,
                station_swap=chosen_station["name"],
                station_bikes=chosen_station["bikes"],
            )
        )
        used_uids.add(chosen_station["uid"])
        current_origin_query = station_query
        current_origin_label = chosen_station["name"]

    raise RuntimeError(
        f"Could not plan a full route within {max_legs} legs. "
        f"Try increasing --max-legs or check that the destination is reachable."
    )


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def print_itinerary(legs: List[Dict]) -> None:
    print("\nWienMobil Rad Route Plan")
    print("=" * 60)

    total_time = 0.0
    total_dist = 0.0

    for leg in legs:
        total_time += leg["duration_min"]
        total_dist += leg["distance_km"]
        print(f"\nLeg {leg['leg']}: {leg['origin_label']}  ->  {leg['destination_label']}")
        print(f"  Duration: {leg['duration_min']:.1f} min   Distance: {leg['distance_km']:.2f} km")
        if leg["station_swap"]:
            bikes_note = f" ({leg['station_bikes']} bikes available now)" if leg["station_bikes"] is not None else ""
            print(f"  >> Dock your bike here and undock a new one{bikes_note}.")
            print(f"     This resets your free-ride window at: {leg['station_swap']}")
        print(f"  Maps link: {leg['maps_url']}")

    swaps = [leg for leg in legs if leg["station_swap"]]
    print("\n" + "-" * 60)
    print(f"Total legs: {len(legs)}   Bike swaps required: {len(swaps)}")
    print(f"Total ride time: {total_time:.1f} min   Total distance: {total_dist:.2f} km")

    if swaps:
        print("\nStations to swap bikes at, in order:")
        for i, leg in enumerate(swaps, 1):
            print(f"  {i}. {leg['station_swap']}")

    print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan a WienMobil Rad (nextbike) bicycle route in Vienna, "
        "automatically inserting bike swaps so no leg exceeds the free-ride limit."
    )
    parser.add_argument("start", help="Start address (e.g. 'Stephansplatz, Wien')")
    parser.add_argument("destination", help="Destination address (e.g. 'Schoenbrunn, Wien')")
    parser.add_argument(
        "--api-key",
        dest="api_key",
        default=None,
        help="Google Maps API key. If omitted, read from GOOGLE_MAPS_API_KEY env var.",
    )
    parser.add_argument(
        "--max-ride-minutes",
        type=float,
        default=DEFAULT_MAX_RIDE_MINUTES,
        help=f"Maximum continuous ride time per leg, in minutes (default: {DEFAULT_MAX_RIDE_MINUTES}).",
    )
    parser.add_argument(
        "--search-minutes",
        type=float,
        default=DEFAULT_SEARCH_MINUTES,
        help=f"Time mark along an over-limit route to search for a transfer station (default: {DEFAULT_SEARCH_MINUTES}).",
    )
    parser.add_argument(
        "--max-legs",
        type=int,
        default=DEFAULT_MAX_LEGS,
        help=f"Safety cap on the number of legs/transfers (default: {DEFAULT_MAX_LEGS}).",
    )
    parser.add_argument(
        "--search-radius-km",
        type=float,
        default=DEFAULT_SEARCH_RADIUS_KM,
        help=f"Max distance to search for a transfer station, in km (default: {DEFAULT_SEARCH_RADIUS_KM}).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    api_key = "AIzaSyANN3VIfYuE_lYqoTDljG_Mieb5othgzrk"
    if not api_key:
        print(
            "Error: no Google Maps API key provided. Pass --api-key or set GOOGLE_MAPS_API_KEY.",
            file=sys.stderr,
        )
        return 1

    try:
        legs = plan_route(
            args.start,
            args.destination,
            api_key,
            max_ride_minutes=args.max_ride_minutes,
            search_minutes=args.search_minutes,
            max_legs=args.max_legs,
            search_radius_km=args.search_radius_km,
        )
    except (RouteNotFoundError, NoStationFoundError, NextbikeAPIError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        return 1

    print_itinerary(legs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
