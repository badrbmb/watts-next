import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import geopandas as gpd
import rasterio
import rasterio.features
import rasterio.io
import rasterio.warp
import structlog
from rasterio.merge import merge as rio_merge
from shapely.geometry.base import BaseGeometry
from tqdm import tqdm

from watts_next.config import DATA_DIR
from watts_next.utils import download_file_from_url, load_geometry_from_geojsons

logger = structlog.getLogger()

OUT_DIR = DATA_DIR / "ghsl"
OUT_DIR.mkdir(exist_ok=True)

# years with data available
REFERENCE_EPOCH = 2020


class GhslRespository:
    def __init__(
        self,
        reference_epoch: int = REFERENCE_EPOCH,
    ) -> None:
        self.reference_epoch = reference_epoch
        self.base_url: str = "https://ghsl.jrc.ec.europa.eu"
        self.tiling_gdf = self._load_tiling_schema()
        self.out_dir = OUT_DIR / f"{reference_epoch}"
        self.out_dir.mkdir(exist_ok=True, parents=True)

    @property
    def tiling_filename(self) -> str:
        """File name of all tilings available."""
        return "GHSL_data_4326_shapefile.zip"

    def _download_tiling_schema(self) -> None:
        """Download the shapefile containing the tiling schema used in the GHSL data."""
        url = f"{self.base_url}/download/{self.tiling_filename}"
        output_path = OUT_DIR / self.tiling_filename
        download_file_from_url(url=url, output_path=output_path)

    def _load_tiling_schema(self) -> gpd.GeoDataFrame:
        """Load the tiling schema df of GHSL data."""
        path = OUT_DIR / self.tiling_filename
        if not path.exists():
            # download the tile file
            logger.debug(
                event="Tiling file not found, downloading ...",
                path=path,
            )
            self._download_tiling_schema()
        gdf = gpd.read_file(path).to_crs("EPSG:4326")
        # keep only valid geometries
        return gdf[gdf["geometry"].apply(lambda x: x.is_valid)].copy()

    def find_intersecting_tile_ids(self, geomety: BaseGeometry) -> list[str]:
        """Return tiles intersecting given geometry."""
        df_tiles = gpd.clip(self.tiling_gdf, geomety)
        return df_tiles["tile_id"].tolist()

    def _get_tile_file_name(self, tile_id: str) -> str:
        return f"GHS_POP_E{self.reference_epoch}_GLOBE_R2023A_4326_3ss_V1_0_{tile_id}.zip"

    def download_tile_data(self, tile_id: str) -> None:
        """Download tile files for a given id across all reference epochs."""
        file_name = self._get_tile_file_name(tile_id=tile_id)
        output_path = self.out_dir / file_name
        if output_path.exists():
            logger.debug(
                event="File already downloaded!",
                tile_id=tile_id,
                path=output_path,
            )
            return
        base_url = self.base_url.replace("ghsl.", "jeodpp.")
        ref_url = f"{base_url}/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A/GHS_POP_E{self.reference_epoch}_GLOBE_R2023A_4326_3ss/V1-0/tiles"  # noqa: E501
        url = f"{ref_url}/{file_name}"
        download_file_from_url(url=url, output_path=output_path)

    def get_merged_tif_path(self, tile_ids: list[str]) -> Path:
        """Get the name of the reference .tif file containing merged infos from all tiles."""
        file_name = hashlib.sha1("".join(sorted(tile_ids)).encode("utf-8")).hexdigest()  # noqa: S324
        return OUT_DIR / f"{self.reference_epoch}_{file_name}.tif"

    def merge_tile_tifs(self, tile_ids: list[str]) -> Path:
        """Merge all tiles into a single file."""
        tif_paths = []
        src_files_to_mosaic = []

        logger.debug(
            event="Extracting individual tile's tif...",
            n_tiles=len(tile_ids),
        )
        with TemporaryDirectory() as temp_dir:
            for tile_id in tqdm(tile_ids):
                path = self.out_dir / self._get_tile_file_name(tile_id)
                with ZipFile(path, "r") as zip_ref:
                    for file_info in zip_ref.infolist():
                        if file_info.filename.endswith(".tif"):
                            tif_path = zip_ref.extract(file_info.filename, path=temp_dir)
                            tif_paths.append(tif_path)
            # Open and collect all .tif files into the list
            for tif_path in tqdm(tif_paths):
                src = rasterio.open(tif_path)
                src_files_to_mosaic.append(src)

            # Merge function returns a single mosaic array and the transformation info
            mosaic, out_trans = rio_merge(src_files_to_mosaic)

        # Save the result as a new .tif file
        out_meta = src_files_to_mosaic[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "compress": "lzw",
            },
        )

        # Path for the output merged file
        output_path = self.get_merged_tif_path(tile_ids=tile_ids)
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        # Close the opened files
        for src in src_files_to_mosaic:
            src.close()

        logger.debug(
            event="Tile tifs merged and written to disk.",
            output_path=output_path,
        )

        return output_path


class GhslService:
    def __init__(self, repository: GhslRespository | None = None) -> None:
        if repository is None:
            repository = GhslRespository()
            logger.debug(
                event="Init. repository with defautl reference epoch.",
                reference_epoch=repository.reference_epoch,
            )
        self.repository = repository

    def create_entsoe_density_population_tif(self, update: bool = False) -> Path:
        """Create a tif file of density population for ENTSO-E zones."""
        zone_paths = list((DATA_DIR / "enstoe/geojson").glob("*.geojson"))
        entsoe_geom = load_geometry_from_geojsons(zone_paths)

        tile_ids = self.repository.find_intersecting_tile_ids(geomety=entsoe_geom)

        merged_tif = self.repository.get_merged_tif_path(tile_ids)

        if merged_tif.exists() and not update:
            logger.info(
                event="Density population .tif file already exists for the chosen region.",
                update=update,
            )
            return merged_tif

        # download individual tile tif
        for tile_id in tile_ids:
            self.repository.download_tile_data(tile_id=tile_id)

        return self.repository.merge_tile_tifs(tile_ids=tile_ids)
