import datetime as dt
from enum import Enum
from typing import Any, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from watts_next.feature_pipeline.base import BaseFeatureGenerator, BasePreProcessor
from watts_next.request import ZoneKey


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

    def load_data(
        self,
        zone_key: ZoneKey,
        start: dt.datetime,
        end: dt.datetime,
        **kwargs: Any,  # noqa: ANN401
    ) -> pd.DataFrame:
        """Load data from source into a Ray Dataset."""
        return self.feature_generator.generate_feature(
            zone_key=zone_key,
            start=start,
            end=end,
            **kwargs,
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
