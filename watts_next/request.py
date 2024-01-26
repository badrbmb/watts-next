from pathlib import Path

import geopandas as gpd
import reverse_geocode
import shapely
from power_stash.inputs.entsoe.request import Area
from pydantic import BaseModel, field_validator
from shapely.geometry.base import BaseGeometry

from watts_next.config import DATA_DIR
from watts_next.utils import load_geometry_from_geojsons, weak_lru

# list of ENTSOE zones to choose from
ENSTOE_ZONE_NAMES = [
    "AT",
    "BE",
    "BG",
    "CH",
    "CZ",
    "DE_LU",
    "DK_1",
    "DK_2",
    "EE",
    "ES",
    "FI",
    "FR",
    "GR",
    "HR",
    "HU",
    "IT_CALA",
    "IT_CNOR",
    "IT_CSUD",
    "IT_NORD",
    "IT_SARD",
    "IT_SICI",
    "IT_SUD",
    "LT",
    "LV",
    "NL",
    "NO_1",
    "NO_2",
    "NO_3",
    "NO_4",
    "NO_5",
    "PL",
    "PT",
    "RO",
    "RS",
    "SE_1",
    "SE_2",
    "SE_3",
    "SE_4",
    "SI",
    "SK",
]


class ZoneKey(BaseModel):
    area: Area

    @field_validator("area")
    def validate_area_allowed(cls, v: Area) -> Area:
        """Validate area is in allowed list."""
        if v.name not in ENSTOE_ZONE_NAMES:
            raise ValueError(f"Expected an aera in allowed list, got area={v.name} instead!")
        return v

    def __hash__(self):  # noqa: ANN204, D105
        return hash(self.area)

    @property
    def geojson_path(self) -> Path:
        """Return the path to local geosjon file defining the boudnary of the zone."""
        return DATA_DIR / f"enstoe/geojson/{self.name}.geojson"

    @property
    def geometry(self) -> BaseGeometry:
        """Return the area geometry."""
        _df = self.get_geo_dataframe()
        return shapely.unary_union(_df["geometry"].tolist())

    def get_geo_dataframe(self) -> gpd.GeoDataFrame:
        """Return the area of the zone key in a geo dataframe."""
        return load_geometry_from_geojsons([self.geojson_path])

    @weak_lru(maxsize=128)
    def reseverse_geocode_country(self) -> dict:
        """Reverse geocode the country of the zone key."""
        lat_lon = [t[::-1] for t in list(self.geometry.centroid.coords)]
        match = reverse_geocode.search(lat_lon)
        if len(match) != 1:
            raise ValueError(
                f"Failed matching country for {self.name}, found {len(match)} matches!",
            )
        return match[0]

    @property
    def country_name(self) -> str:
        """Return the country name where the zone is located."""
        country = self.reseverse_geocode_country()
        return country["country"]

    @property
    def country_iso2(self) -> str:
        """Return the country iso2 where the zone is located."""
        country = self.reseverse_geocode_country()
        return country["country_code"]

    @property
    def name(self) -> str:
        """Return area name."""
        return self.area.name
