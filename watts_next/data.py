import datetime as dt
from enum import Enum
from typing import Any, Iterator

import numpy as np
import pandas as pd
import structlog
from sklearn.model_selection import TimeSeriesSplit

from watts_next.feature_pipeline.base import BaseFeatureGenerator, BasePreProcessor
from watts_next.request import ZoneKey

logger = structlog.get_logger()


class FeatureType(Enum):
    AGGREGATE = "aggregate"


class PreProcessorType(Enum):
    CALENDAR = "calendar"


class DataLoader:
    def __init__(
        self,
        feature_type: FeatureType,
        preprocessor_type: PreProcessorType,
        feature_generator_args: dict | None = None,
        preprocessor_args: dict | None = None,
    ) -> None:
        super().__init__()
        # assing the feature_generator
        self.feature_generator = self._load_feature_generator(
            feature_type,
            feature_generator_args or {},
        )
        # assing the pre-processor
        self.preprocessor = self._load_preprocessor(
            preprocessor_type,
            preprocessor_args or {},
        )

    @staticmethod
    def _load_feature_generator(
        feature_type: FeatureType,
        feature_generator_args: dict,
    ) -> BaseFeatureGenerator:
        match feature_type:
            case FeatureType.AGGREGATE:
                from watts_next.feature_pipeline.generator import AggregateFeatureGenerator

                feature_generator = AggregateFeatureGenerator(
                    **feature_generator_args,
                )
            case _:
                raise NotImplementedError(f"{feature_type=} not implemented!")

        return feature_generator

    @staticmethod
    def _load_preprocessor(
        preprocessor_type: PreProcessorType,
        preprocessor_args: dict,
    ) -> BasePreProcessor:
        match preprocessor_type:
            case PreProcessorType.CALENDAR:
                from watts_next.feature_pipeline.processor import CalendarPreprocessor

                preprocessor = CalendarPreprocessor(
                    **preprocessor_args,
                )
            case _:
                raise NotImplementedError(f"{preprocessor_type=} not implemented!")

        return preprocessor

    def generate_features(
        self,
        zone_key: ZoneKey,
        start: dt.datetime,
        end: dt.datetime,
        **kwargs: Any,  # noqa: ANN401
    ) -> pd.DataFrame:
        """Generate features using the feature_generator."""
        logger.debug(
            event="Generating features ...",
            feature_generator=type(self.feature_generator).__name__,
            zone_key=zone_key,
            start=start,
            end=end,
        )
        return self.feature_generator.generate_feature(
            zone_key=zone_key,
            start=start,
            end=end,
            **kwargs,
        )

    def transform(self, df: pd.DataFrame, zone_key: ZoneKey) -> pd.DataFrame:
        """Transform the features using the preprocessor."""
        logger.debug(
            event="Transforming features ...",
            preprocessor=type(self.preprocessor).__name__,
            zone_key=zone_key,
        )
        return self.preprocessor.transform(df=df, zone_key=zone_key)

    def load_data(
        self,
        zone_key: ZoneKey,
        start: dt.datetime,
        end: dt.datetime,
        **kwargs: Any,  # noqa: ANN401
    ) -> pd.DataFrame:
        """Generate features are transform them.

        kwargs are passed to the feature_generator only!
        """
        return self.transform(
            df=self.generate_features(zone_key, start, end, **kwargs),
            zone_key=zone_key,
        )

    @staticmethod
    def get_train_test_indices(
        df: pd.DataFrame,
        n_splits: int = 5,
        gap: int = 0,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Splits a dataset with timeseries features into train and test sets."""
        tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        return tss.split(df)
