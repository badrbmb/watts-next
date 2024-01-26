import datetime as dt
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from watts_next.request import ZoneKey


class BasePreProcessor(ABC):
    @abstractmethod
    def __init__(self, *args: Any) -> None:  # noqa: ANN401
        pass

    @abstractmethod
    def fit(self, *args: Any) -> None:  # noqa: ANN401
        """Abstract fit method."""
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame, zone_key: ZoneKey) -> pd.DataFrame:
        """Abstract transform method."""
        pass

    @staticmethod
    def add_hour(df: pd.DataFrame) -> pd.DataFrame:
        """Adds Day feature to df."""
        df["hour"] = [t.hour for t in df.index]
        return df


class BaseFeatureGenerator(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate_feature(
        self,
        zone_key: ZoneKey,
        start: dt.datetime,
        end: dt.datetime,
    ) -> pd.DataFrame:
        """Generate features."""
        pass
