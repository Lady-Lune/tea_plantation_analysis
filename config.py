"""Shared configuration and pipeline contracts from the GUIDE."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MANIFESTS_DIR = DATA_DIR / "manifests"
SRI_LANKA_PATH = DATA_DIR / "sri_lanka.gpkg"
ROADS_PATH = DATA_DIR / "roads.gpkg"
HEALTH_FACILITIES_PATH = DATA_DIR / "health_facilities.gpkg"
SCHOOLS_PATH = DATA_DIR / "schools_by_tier.gpkg"
STUDY_AREA_PATH = DATA_DIR / "study_area.gpkg"
GRAPH_PATH = DATA_DIR / "graph_multimodal.graphml"
TIN_GRAPH_PATH = DATA_DIR / "graph_multimodal_tin.graphml"
ACCESSIBILITY_SCORES_PATH = DATA_DIR / "accessibility_scores.csv"
STORAGE_CRS = "EPSG:4326"
METRIC_CRS = "EPSG:5235"
STUDY_AREA_BUFFER_M = 15_000


@dataclass(frozen=True)
class ArcGISLayerConfig:
    """Configuration for one paginated ArcGIS FeatureServer layer."""

    name: str
    service_url: str
    layer_id: int


@dataclass(frozen=True)
class FacilityTier:
    """A supply tier and its network-cost threshold."""

    code: str
    label: str
    max_cost_m: int


def arcgis_layers() -> tuple[ArcGISLayerConfig, ...]:
    """Return the GN divisions and tea-district service definitions."""
    return (
        ArcGISLayerConfig(
            "gn_divisions",
            "https://services5.arcgis.com/MbTeS0GdIaAHUwfD/arcgis/rest/services/GN_Divisions_Sri_Lanka/FeatureServer",
            0,
        ),
        ArcGISLayerConfig(
            "tea_districts",
            "https://services1.arcgis.com/0MSEUqKaxRlEPj5g/arcgis/rest/services/Tea_growing_districts/FeatureServer",
            0,
        ),
    )


def school_tiers() -> tuple[FacilityTier, ...]:
    """Return the four school tiers from the canonical YAML specification."""
    return (
        FacilityTier("1AB", "national_secondary_AL_science", 5_000),
        FacilityTier("1C", "provincial_secondary_AL_arts", 5_000),
        FacilityTier("Type2", "secondary_up_to_OL", 3_000),
        FacilityTier("Type3", "primary_only", 2_000),
    )


def health_tiers() -> tuple[FacilityTier, ...]:
    """Return the five health tiers from the canonical YAML specification."""
    return (
        FacilityTier("estate_dispensary", "estate_dispensary", 2_000),
        FacilityTier("phc", "phc", 5_000),
        FacilityTier("divisional_hospital", "divisional_hospital", 10_000),
        FacilityTier("base_hospital", "base_hospital", 15_000),
        FacilityTier("teaching_hospital", "teaching_hospital", 30_000),
    )
