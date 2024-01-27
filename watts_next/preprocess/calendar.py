from functools import lru_cache

import holidays
import pandas as pd
from holidays.holiday_base import HolidayBase
from sklearn.base import BaseEstimator, TransformerMixin
from sktime.transformations.series.date import DateTimeFeatures

DEFAULT_CALENDAR_SELECTION = [
    "year",
    "month_of_year",
    "week_of_year",
    "day_of_year",
    "day_of_month",
    "day_of_week",
    "hour_of_day",
    "is_weekend",
]


@lru_cache
def _get_country_holidays(country_iso2: str) -> HolidayBase:
    return holidays.country_holidays(
        country=country_iso2,
    )


class CalendarPreprocessor(BaseEstimator, TransformerMixin):
    """Custom preprocessor class adding calendar features to train on."""

    def __init__(
        self,
        calendar_selection: list[str] = DEFAULT_CALENDAR_SELECTION,
        include_holidays: bool = True,
        country_iso2: str | None = None,
    ) -> None:
        self.calendar_selection = calendar_selection
        self.include_holidays = include_holidays
        # init. DateTimeFeatures
        self.calendar_features = DateTimeFeatures(
            manual_selection=self.calendar_selection,
            keep_original_columns=True,
        )
        self._country_iso2 = country_iso2

    @property
    def zone_holidays(self) -> HolidayBase | None:
        """Get the zone holidays."""
        return _get_country_holidays(self.country_iso2) if self.country_iso2 is not None else None

    @property
    def country_iso2(self) -> str | None:
        """Country ISO2."""
        return self._country_iso2

    @country_iso2.setter
    def country_iso2(self, value: str) -> None:
        self._country_iso2 = value

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CalendarPreprocessor":  # noqa: ARG002
        """Fit method for the preprocessor."""
        # the default fit of sktime, nothing happens.
        self.calendar_features.fit(X)
        return self

    def flag_holiday(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag holidays in df."""
        df["is_holiday"] = df.index.isin(self.zone_holidays).astype(int)
        return df

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """Transform dataframe by adding new calendar features."""
        df = X.copy()
        if self.include_holidays:
            df = self.flag_holiday(df)
        df = self.calendar_features.transform(X, y)
        return df
