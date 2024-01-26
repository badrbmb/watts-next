import functools
import weakref
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import pandas as pd
import requests


def weak_lru(
    maxsize: int = 128,
    typed: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """LRU Cache decorator that keeps a weak reference to "self"."""

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.lru_cache(maxsize, typed)
        def _func(_self: weakref.ref, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            # Dereference the weakref and call the original function
            return func(_self(), *args, **kwargs)

        @functools.wraps(func)
        def inner(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            # Create a weakref to 'self' and call the cached function
            return _func(weakref.ref(self), *args, **kwargs)

        return inner

    return wrapper


def download_file_from_url(url: str, output_path: Path) -> None:
    """Download file from url and store at desired path."""
    # download the file from this url to a tmp location @ tmp_path
    response = requests.get(url=url, stream=True, timeout=20)
    response.raise_for_status()
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


def load_geometry_from_geojsons(paths: list[Path]) -> gpd.GeoDataFrame:
    """Load all geosjon files and return unary union."""
    geo_df = pd.concat(
        [gpd.read_file(path) for path in paths],
    )
    geo_df = geo_df.to_crs("EPSG:4326")  # type: ignore
    return geo_df
