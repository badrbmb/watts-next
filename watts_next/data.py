import datetime as dt
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sklearn.pipeline import Pipeline
from sktime.split import TemporalTrainTestSplitter

from watts_next.feature_pipeline.base import BaseFeatureGenerator
from watts_next.request import ZoneKey

logger = structlog.get_logger()


class DataLoader:
    def __init__(
        self,
        feature_generator: BaseFeatureGenerator,
        preprocessing_pipeline: Pipeline,
    ) -> None:
        self.feature_generator = feature_generator
        self.preprocessing_pipeline = preprocessing_pipeline

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

    @property
    def preprocessing_pipeline_steps(self) -> list[str]:
        """Get preprocessing pipeline step names."""
        return [type(t[1]).__name__ for t in self.preprocessing_pipeline.steps]

    def _assign_preprocessing_property(self, property_name: str, value: Any) -> None:  # noqa: ANN401
        """Helper function to assign property to pre-processing steps."""
        for _, step_model in self.preprocessing_pipeline.steps:
            if hasattr(step_model, property_name):
                setattr(step_model, property_name, value)

    def transform(self, df: pd.DataFrame, zone_key: ZoneKey) -> pd.DataFrame:
        """Transform the features using the preprocessor."""
        logger.debug(
            event="Transforming features ...",
            preprocessing_pipeline=self.preprocessing_pipeline_steps,
            zone_key=zone_key,
        )
        self._assign_preprocessing_property(
            property_name="country_iso2",
            value=zone_key.country_iso2,
        )
        return self.preprocessing_pipeline.fit_transform(X=df, y=None)

    def load_data(
        self,
        zone_key: ZoneKey,
        start: dt.datetime,
        end: dt.datetime,
        **kwargs: Any,  # noqa: ANN401
    ) -> pd.DataFrame:
        """Generate features + transform them.

        kwargs are passed to the feature_generator only.
        """
        return self.transform(
            df=self.generate_features(zone_key, start, end, **kwargs),
            zone_key=zone_key,
        )

    @staticmethod
    def get_train_test_indices(
        df: pd.DataFrame,
        test_size: float = 0.3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Splits a dataset with timeseries features into train and test sets."""
        splitter = TemporalTrainTestSplitter(test_size=test_size)
        return next(splitter.split(df.index))

    @staticmethod
    def get_train_test_data(
        df: pd.DataFrame,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        target_column: str = "value",
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Return test and train features and targets from a df using a list of indices."""
        df_train = df.iloc[train_indices]
        df_test = df.iloc[test_indices]

        X_train = df_train[[t for t in df_train.columns if t != target_column]].copy()
        y_train = df_train[target_column].copy()

        X_test = df_test[[t for t in df_test.columns if t != target_column]].copy()
        y_test = df_test[target_column].copy()

        return X_train, y_train, X_test, y_test
