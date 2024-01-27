import datetime as dt
from abc import ABC, abstractmethod

import pandas as pd

from watts_next.request import ZoneKey


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
