from typing import Any

import holidays
import pandas as pd

from watts_next.feature_pipeline.base import BasePreProcessor
from watts_next.request import ZoneKey


class CalendarPreprocessor(BasePreProcessor):
    """Custom preprocessor class."""

    def __init__(
        self,
        include_calendar_features: bool = True,
        include_holidays: bool = True,
    ) -> None:
        self.include_calendar_features = include_calendar_features
        self.include_holidays = include_holidays

    def fit(self, *args: Any) -> "CalendarPreprocessor":  # noqa: ANN401, ARG002
        """Nothing to fit."""
        return self

    @staticmethod
    def add_day(df: pd.DataFrame) -> pd.DataFrame:
        """Adds Day feature to df."""
        df["day"] = [t.day for t in df.index]
        return df

    @staticmethod
    def flag_weekend(df: pd.DataFrame) -> pd.DataFrame:
        """Flag weekend days."""
        df["is_weekend"] = [t.day_of_week in [5, 6] for t in df.index]
        return df

    @staticmethod
    def add_week(df: pd.DataFrame) -> pd.DataFrame:
        """Adds Week number feature to df."""
        df["week"] = [t.week for t in df.index]
        return df

    @staticmethod
    def add_month(df: pd.DataFrame) -> pd.DataFrame:
        """Adds Month feature to df."""
        df["month"] = [t.month for t in df.index]
        return df

    @staticmethod
    def add_holidays(df: pd.DataFrame, zone_key: ZoneKey) -> pd.DataFrame:
        """Flag holidays in df."""
        df_out = df.copy()
        zone_holidays = holidays.country_holidays(
            country=zone_key.country_iso2,
            years={t.year for t in df.index},
        )
        df_out["holiday"] = df_out.apply(lambda x: zone_holidays.get(x.name, "NA"), axis=1)
        return df_out

    def add_base_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all base calendar features to df."""
        df_out = df.copy()
        df_out = (
            df_out.pipe(self.add_hour)
            .pipe(self.add_day)
            .pipe(self.flag_weekend)
            .pipe(self.add_week)
            .pipe(self.add_month)
        )
        return df_out

    def transform(self, df: pd.DataFrame, zone_key: ZoneKey) -> pd.DataFrame:
        """Transform dataframe by adding new calendar features."""
        if self.include_calendar_features:
            # add base calendar features
            df = self.add_base_calendar_features(df)
        if self.include_holidays:
            # add holidays
            df = self.add_holidays(df=df, zone_key=zone_key)

        return df
