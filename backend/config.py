from pathlib import Path
from typing import Dict, Any, Optional
import osmnx as ox

EARTH_RADIUS_KM = 6371.0

OSM_CACHE_DIR = Path("cache/osm")
OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# OSMnx HTTP + file caching
ox.settings.use_cache = True
ox.settings.log_console = False

RUN_PROFILES: Dict[str, Dict[str, Any]] = {
    "default_balanced": {
        "preview_initial_radius_km": 12.0,
        "preview_max_radius_km": 30.0,
        "preview_local_cap_km": 25.0,
        "preview_osm_threshold_km": 25.0,
        "preview_max_total_stops": 20,
        "enhanced_fairness_weight": 0.60,
        "enhanced_distance_weight": 0.25,
        "enhanced_time_weight": 0.15,
        "enhanced_max_iterations": 20,
        "enhanced_border_fraction": 0.50,
    },
    "amazon_expanded_search": {
        "preview_initial_radius_km": 25.0,
        "preview_max_radius_km": 120.0,
        "preview_local_cap_km": 100.0,
        "preview_osm_threshold_km": 35.0,
        "preview_max_total_stops": 60,
        "enhanced_fairness_weight": 0.35,
        "enhanced_distance_weight": 0.45,
        "enhanced_time_weight": 0.20,
        "enhanced_max_iterations": 20,
        "enhanced_border_fraction": 0.35,
    },
    "zomato_expanded_search": {
        "preview_initial_radius_km": 20.0,
        "preview_max_radius_km": 60.0,
        "preview_local_cap_km": 50.0,
        "preview_osm_threshold_km": 35.0,
        "preview_max_total_stops": 40,
        "enhanced_fairness_weight": 0.60,
        "enhanced_distance_weight": 0.25,
        "enhanced_time_weight": 0.15,
        "enhanced_max_iterations": 20,
        "enhanced_border_fraction": 0.50,
    },
}

DEMO_PREVIEW_DEPOTS: Dict[str, Optional[str]] = {
    "primary_reconstruction": "DEPOT-130",
    "comparative_template": "DEPOT-153",
    "generic_uploaded_dataset": None,
}

MIN_FIXED_DEMO_NODES = 12
MIN_FIXED_DEMO_AGENTS = 6
AMAZON_FIXED_DEMO_NODES = 0
AMAZON_FIXED_DEMO_AGENTS = 0
AMAZON_DEFAULT_REPRESENTATIVES = 6
AMAZON_MAX_CUSTOMERS_PER_REP = 3
AMAZON_MIN_PREVIEW_STOPS = AMAZON_DEFAULT_REPRESENTATIVES * AMAZON_MAX_CUSTOMERS_PER_REP
