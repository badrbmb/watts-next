import geopandas as gpd
import structlog
from shapely.geometry.base import BaseGeometry

from watts_next.config import DATA_DIR
from watts_next.utils import download_file_from_url

logger = structlog.getLogger()

OUT_DIR = DATA_DIR / "ghsl"

# years with data available
REFERENCE_EPOCH = 2020

REFERENCE_RESOLUTION = 100  # meters


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
        return "GHSL_data_54009_shapefile.zip"

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
        return gpd.read_file(path).to_crs("EPSG:4326")

    def find_intersecting_tile_ids(self, geomety: BaseGeometry) -> list[str]:
        """Return tiles intersecting given geometry."""

        def _intersects(geom: BaseGeometry) -> bool:
            try:
                return geomety.intersects(geom)
            except Exception:
                # TODO: refine exception handling for cases liek below:
                # IllegalArgumentException: CGAlgorithmsDD::orientationIndex encountered NaN/Inf numbers
                return False

        df_tiles = self.tiling_gdf.copy()
        df_tiles = df_tiles[df_tiles["geometry"].apply(lambda x: _intersects(x))].copy()
        return df_tiles["tile_id"].tolist()

    def _get_tile_file_name(self, tile_id: str) -> str:
        return f"GHS_BUILT_S_E{self.reference_epoch}_GLOBE_R2023A_54009_100_V1_0_{tile_id}.zip"

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
        ref_url = f"{base_url}/ftp/jrc-opendata/GHSL/GHS_BUILT_S_GLOBE_R2023A/GHS_BUILT_S_E{self.reference_epoch}_GLOBE_R2023A_54009_100/V1-0/tiles"
        url = f"{ref_url}/{file_name}"
        download_file_from_url(url=url, output_path=output_path)
