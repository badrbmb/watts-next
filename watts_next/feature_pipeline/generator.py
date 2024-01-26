import datetime as dt
from pathlib import Path

import dask.dataframe
import pandas as pd
import structlog
from power_stash.inputs.entsoe import models
from power_stash.inputs.entsoe.request import RequestType
from power_stash.outputs.database.repository import SqlRepository
from sqlalchemy import and_, select

from watts_next.feature_pipeline.base import BaseFeatureGenerator
from watts_next.feature_pipeline.ghsl import GhslRespository
from watts_next.request import ZoneKey

logger = structlog.getLogger()


class AggregateFeatureGenerator(BaseFeatureGenerator):
    """Aggreates all NWP data into a single statistics by variable and timestamp."""

    def __init__(
        self,
        path_to_nwp_files: Path,
        sql_repository: SqlRepository | None = None,
        density_weighted: bool = False,
    ) -> None:
        self.path_to_nwp_files = path_to_nwp_files
        self.density_weighted = density_weighted
        if sql_repository is None:
            sql_repository = SqlRepository()
        self.sql_repository = sql_repository
        if self.density_weighted:
            ghsl_repository = GhslRespository()
            logger.debug(
                event="Init. GHSL repository with default reference epoch.",
                reference_epoch=ghsl_repository.reference_epoch,
            )
            # use for density weighted calculations
            self.ghsl_repository = ghsl_repository
        else:
            self.ghsl_repository = None

    @staticmethod
    def create_density_population_tif(
        ghsl_repository: GhslRespository,
        zone_key: ZoneKey,
        update: bool = False,
    ) -> Path:
        """Create a tif file of density population for ENTSO-E zone key."""
        tile_ids = ghsl_repository.find_intersecting_tile_ids(geomety=zone_key.geometry)

        merged_tif = ghsl_repository.get_merged_tif_path(tile_ids)

        if merged_tif.exists() and not update:
            logger.info(
                event="Density population .tif file already exists for the zone.",
                update=update,
                zone_key=zone_key.name,
            )
            return merged_tif

        # download individual tile tif
        for tile_id in tile_ids:
            ghsl_repository.download_tile_data(tile_id=tile_id)

        return ghsl_repository.merge_tile_tifs(tile_ids=tile_ids)

    @staticmethod
    def _load_files(path: Path, extension: str) -> list[Path]:
        paths = path.glob(f"**/*.{extension}")
        return sorted(t for t in paths if t.is_file())

    def extract_weather_data(
        self,
        zone_key: ZoneKey,
        start: dt.datetime,
        end: dt.datetime,
    ) -> dask.dataframe.DataFrame:
        """Load NWP data from path."""
        parquet_files = self._load_files(self.path_to_nwp_files, extension="parquet")
        return dask.dataframe.read_parquet(
            parquet_files,
            filters=[
                ("country_name", "==", zone_key.country_name),
                ("timestamp", ">=", start),
                ("timestamp", "<=", end),
            ],
        )

    def transform_weather_data(
        self,
        ddf: dask.dataframe.DataFrame,
    ) -> pd.DataFrame:
        """Transform weather data."""
        if self.density_weighted:
            # density weighting NWP variables not worth the computation effort (c.f explo. notebook)
            raise NotImplementedError("Density weighted variables not implemented!")
        ddf_averages_by_run = (
            ddf.groupby(["run_time", "timestamp"])
            .agg({"u10": "mean", "v10": "mean", "t2m": "mean", "tp": "mean"})
            .reset_index()
        )
        ddf_averages_by_run = ddf_averages_by_run.sort_values(["run_time", "timestamp"])
        ddf_averages = ddf_averages_by_run.drop_duplicates(subset="timestamp", keep="last")
        # smooth out tp due to oper having different sampling timestamp in open data version.
        ddf_averages["tp"] = ddf_averages["tp"].rolling(2).mean().fillna(method="bfill")
        df: pd.DataFrame = ddf_averages.compute()
        # timezone for NPW data is UTC
        df["timestamp"] = df["timestamp"].dt.tz_localize(
            "UTC",
            ambiguous="infer",
        )
        df = df[["timestamp", "u10", "v10", "t2m", "tp"]].set_index("timestamp")
        return df

    def extract_electricity_data(
        self,
        request_type: RequestType,
        zone_key: ZoneKey,
        start: dt.datetime,
        end: dt.datetime,
    ) -> pd.DataFrame:
        """Load electricity data."""
        match request_type:
            case RequestType.CONSUMPTION:
                table_model = models.EntsoeHourlyConsumption
            case _:
                raise NotImplementedError(f"{request_type} not implemented!")

        query = (
            select(table_model.timestamp, table_model.value)
            .where(table_model.area == zone_key.area)
            .where(and_(start <= table_model.timestamp, table_model.timestamp <= end))
            .order_by(table_model.timestamp)
        )
        df: pd.DataFrame = self.sql_repository.query(statement=query, return_df=True)
        # timezone for electricity data is UTC
        df["timestamp"] = df["timestamp"].dt.tz_localize(
            "UTC",
            ambiguous="infer",
        )
        df = df.set_index("timestamp")[["value"]]
        return df

    def generate_feature(
        self,
        zone_key: ZoneKey,
        start: dt.datetime,
        end: dt.datetime,
        request_type: RequestType = RequestType.CONSUMPTION,
        add_target: bool = True,
    ) -> pd.DataFrame:
        """Create features and target for given zone key."""
        # load NWP data (features)
        df_data = self.transform_weather_data(
            ddf=self.extract_weather_data(
                zone_key=zone_key,
                start=start,
                end=end,
            ),
        )
        if add_target:
            # Load consumption data (target)
            df_electricity_data = self.extract_electricity_data(
                request_type=request_type,
                zone_key=zone_key,
                start=start,
                end=end,
            )
            # join the two
            df_data = df_electricity_data.join(df_data)

            # fill missing value (3h NWP frequency vs hourly electricyt data)
            df_data = df_data.interpolate(method="linear", limit=3)
        return df_data
