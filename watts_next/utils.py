from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
import shapely
from shapely.geometry.base import BaseGeometry


def download_file_from_url(url: str, output_path: Path) -> None:
    """Download file from url and store at desired path."""
    # download the file from this url to a tmp location @ tmp_path
    response = requests.get(url=url, stream=True, timeout=20)
    response.raise_for_status()
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


def load_geometry_from_geojsons(paths: list[Path]) -> BaseGeometry:
    """Load all geosjon files and return unary union."""
    geo_df = pd.concat(
        [gpd.read_file(path) for path in paths],
    )
    geo_df = geo_df.to_crs("EPSG:4326")  # type: ignore

    return shapely.unary_union(geo_df["geometry"].tolist())
