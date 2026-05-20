# Backend — File Reference

This document provides a formal, concise description of each source file and directory contained in the `backend` folder.

---

## Core Backend Files

### **app.py**

FastAPI application entrypoint and top-level orchestration. Implements API endpoints, dataset normalization and reconstruction routing, model training helpers, and delegates routing, enhancement and OSM operations to service modules.

---

### **amazon.py**

Dataset-specific reconstruction and preview utilities for the primary (Amazon-style) dataset. Provides order-level routing row builders, dataset reconstruction from raw uploads, preview selection, and Amazon-specific assignment/polishing routines.

---

### **zomato.py**

Dataset-specific reconstruction utilities for the Zomato comparative template dataset. Normalises differing column names, infers agent identifiers, and produces a cleaned routing-ready schema compatible with downstream services.

---

### **config.py**

Static configuration constants and runtime settings used across the backend (run profiles, demo preview depot IDs, thresholds, and other feature flags). Centralises tunable parameters for reproducible runs.

---

### **data_preprocessing_utils.py**

Data-cleaning and preprocessing helper functions used prior to reconstruction and normalization. Contains transformations, parsing utilities, and lightweight validation helpers for raw uploads.

---

### **helpers.py**

Shared utility library for common operations:

- Geographic calculations:
  - `haversine_km`
  - `road_adjusted_km`

- Mapping and reconstruction primitives:
  - `FieldMapping`
  - `_base_reconstruct_from_mapping`

- Preview selection helpers
- Dataset inference utilities
- Lightweight validation routines

---

### **schemas.py**

Pydantic models describing request and response payloads and internal schemas (e.g., `BaselineRequest`, `EnhancedRequest`, mapping models). Enforces typed validation at API boundaries.

---

# Runtime Directories

### **artifacts/** _(directory)_

Generated output artifacts such as analysis plots and diagnostics (PNG images under `feature_distributions`). These are runtime outputs and should be treated as build artifacts, not source code.

---

### **cache/** _(directory)_

Local caching of computed objects and external data (for example, OSM caches and JSON blobs). Cache files speed up repeated runs but are not required in source control.

---

# Service Layer (`services/`)

### **services/**init**.py**

Package initializer for the `services` package; may also expose convenience imports for commonly used service functions.

---

### **services/add_customer_service.py**

Handles processing of user-added customers, validation of inputs, and logic to append or assign newly added customer rows to an existing preview/assignment DataFrame prior to routing.

---

### **services/enhancement_service.py**

Implements assignment enhancement algorithms and evaluation routines. Contains optimization, local moves, swaps, and polish functions used to improve baseline assignments.

---

### **services/metrics_service.py**

KPI and statistical utilities including fairness metrics, workload balance index, rep summary card generation, and other key performance indicator calculations consumed by the frontend and enhancement logic.

---

### **services/osm_service.py**

OpenStreetMap integration helpers:

- Building preview graphs with `osmnx`
- Snapping preview points to the road network
- Creating distance/time matrices
- Path coordinate extraction
- Display geometry assembly for route visualization

---

### **services/routing_service.py**

Core routing orchestration responsible for:

- Routing a single representative
- Routing all representatives
- Computing ordered stops from assignment frames
- Evaluating route leg metrics

This module bridges assignment DataFrames with distance matrices and enhancement outputs.

---

# Notes

- The three reconstruction implementations:
  - `amazon.py`
  - `zomato.py`
  - Generic reconstruction paths in `app.py`

share considerable logic for:

- Node creation
- Distance calculations
- Basic normalization

Repeated reconstruction post-processing steps have been extracted into `helpers.py` as `finalize_reconstructed_dataset()` to reduce duplication and improve maintainability. The callers updated include `amazon.py`, `zomato.py`, and the generic reconstruction path in `app.py`.

---

- `__pycache__` directories are generated automatically by Python on import.

Disabling bytecode writes:

```bash
PYTHONDONTWRITEBYTECODE=1
```
