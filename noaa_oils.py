"""
noaa_oils.py — Grounded NOAA ADIOS Oil Database Lookup

Downloads (once) and parses the live NOAA ADIOS database (the same source
OpenDrift/OpenOil queries), mapping each oil to one of the three SAR
thickness classes using NOAA's own official labels.

Usage:
    from noaa_oils import get_real_oil_for_class, select_openoil_type_grounded

    result = select_openoil_type_grounded("Thick_Emulsified", region_keyword="Gulf")
    print(result["real_oil_name"], result["api_gravity"])
"""

import os
import json
import glob
import shutil
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# NOAA ADIOS Oil Category → Thickness Class Mapping
# (Uses NOAA's own official label vocabulary, not API-gravity cutoffs)
# ---------------------------------------------------------------------------
LIGHT_LABELS  = {"Light Crude", "Condensate", "Gasoline", "Kerosene", "Jet Fuel"}
MEDIUM_LABELS = {"Medium Crude", "Diesel", "Gas Oil", "No. 2 Fuel Oil"}
HEAVY_LABELS  = {
    "Heavy Crude", "Bunker C", "HFO", "IFO",
    "Heavy Fuel Oil", "Bitumen", "Residual Fuel", "No. 6 Fuel Oil"
}

# Generic OpenOil placeholders used when no ADIOS match is found
GENERIC_OIL_TYPES = {
    "Thin_Sheen":       "GENERIC LIGHT CRUDE",
    "Moderate":         "GENERIC MEDIUM CRUDE",
    "Thick_Emulsified": "GENERIC HEAVY CRUDE",
}

# Local cache location (stored alongside this module)
_MODULE_DIR  = os.path.dirname(os.path.abspath(__file__))
_ADIOS_DIR   = os.path.join(_MODULE_DIR, "noaa-oil-data")
_ADIOS_REPO  = "https://github.com/OpenDrift/noaa-oil-data.git"

# In-memory cache so we only parse JSON once per process
_adios_cache: dict | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_adios_cloned() -> bool:
    """Clone the NOAA ADIOS repo shallowly (once).  Returns True on success."""
    if os.path.isdir(os.path.join(_ADIOS_DIR, "data")):
        return True  # already present

    git_exe = shutil.which("git")
    if git_exe is None:
        return False  # git not available

    try:
        subprocess.run(
            [git_exe, "clone", "--depth", "1", _ADIOS_REPO, _ADIOS_DIR],
            check=True,
            capture_output=True,
            timeout=120,
        )
        return True
    except Exception:
        return False


def _load_adios() -> dict:
    """Parse all ADIOS JSON files and bucket them into the 3 thickness classes."""
    global _adios_cache

    if _adios_cache is not None:
        return _adios_cache

    result = {"Thin_Sheen": [], "Moderate": [], "Thick_Emulsified": []}

    if not _ensure_adios_cloned():
        # Graceful degradation: return empty buckets
        _adios_cache = result
        return _adios_cache

    pattern = os.path.join(_ADIOS_DIR, "data", "oil", "**", "*.json")
    for f in glob.glob(pattern, recursive=True):
        try:
            record = json.load(open(f, encoding="utf-8"))
            meta   = record.get("metadata", {})
            labels = set(meta.get("labels", []))
            entry  = {
                "name":     meta.get("name", "Unknown"),
                "location": meta.get("location", "Unknown"),
                "api":      meta.get("API"),
                "labels":   sorted(labels),
            }
            if labels & LIGHT_LABELS:
                result["Thin_Sheen"].append(entry)
            elif labels & MEDIUM_LABELS:
                result["Moderate"].append(entry)
            elif labels & HEAVY_LABELS:
                result["Thick_Emulsified"].append(entry)
        except Exception:
            continue

    _adios_cache = result
    return _adios_cache


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_real_oil_for_class(thickness_class: str, region_keyword: str | None = None) -> dict | None:
    """
    Return a real, NOAA-documented oil entry for the given thickness class.

    Args:
        thickness_class:  One of "Thin_Sheen", "Moderate", "Thick_Emulsified".
        region_keyword:   Optional substring to match against oil location
                          (e.g. "Gulf", "Norway", "Saudi Arabia").

    Returns:
        A dict with keys: name, location, api, labels — or None if no match.
    """
    db = _load_adios()
    candidates = db.get(thickness_class, [])

    if region_keyword:
        filtered = [c for c in candidates if region_keyword.lower() in c["location"].lower()]
        if filtered:
            candidates = filtered

    return candidates[0] if candidates else None


def select_openoil_type_grounded(
    thickness_class: str,
    region_keyword: str | None = None,
) -> dict:
    """
    Return a real, NOAA-documented oil recommendation (with citable API
    gravity + source location) for the given thickness class.

    Falls back to the generic OpenOil placeholder if the ADIOS database is
    unavailable or contains no match for the requested class/region.

    Args:
        thickness_class:  One of "Thin_Sheen", "Moderate", "Thick_Emulsified".
        region_keyword:   Optional region filter string.

    Returns:
        A dict containing:
            thickness_class, real_oil_name, api_gravity, source_location,
            noaa_labels, opendrift_oiltype, fallback_generic_type
        — or a degraded dict with only opendrift_oiltype + notes on failure.
    """
    real_oil = get_real_oil_for_class(thickness_class, region_keyword)
    generic  = GENERIC_OIL_TYPES.get(thickness_class, "GENERIC CRUDE")

    if real_oil:
        return {
            "thickness_class":      thickness_class,
            "real_oil_name":        real_oil["name"],
            "api_gravity":          real_oil["api"],
            "source_location":      real_oil["location"],
            "noaa_labels":          real_oil["labels"],
            # OpenOil can look this oil up by name directly via its built-in ADIOS client
            "opendrift_oiltype":    real_oil["name"],
            "fallback_generic_type": generic,
        }

    return {
        "thickness_class":   thickness_class,
        "opendrift_oiltype": generic,
        "notes": "No real ADIOS match found locally — using generic OpenOil placeholder.",
    }


def get_adios_summary() -> dict:
    """Return a count of catalogued oils per class (useful for diagnostics)."""
    db = _load_adios()
    return {cls: len(entries) for cls, entries in db.items()}
