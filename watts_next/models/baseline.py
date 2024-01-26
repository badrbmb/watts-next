from enum import Enum

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

pd.offsets.Day()

NUM_DAYS_IN_MONTH = 30
NUM_DAYS_IN_YEAR = 365


class ForecastType(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class AheadHourlyForecastTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, forecast_type: ForecastType | str) -> None:
        if isinstance(forecast_type, str):
            forecast_type = ForecastType(forecast_type)
        self.forecast_type = forecast_type

    def fit(
        self,
        X: pd.DataFrame,  # noqa: ARG002
        y: pd.DataFrame | None = None,  # noqa: ARG002
    ) -> "AheadHourlyForecastTransformer":
        """Nothing to fit."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict based on foresting type."""
        match self.forecast_type:
            case ForecastType.HOURLY:
                periods = 1
                freq = "h"
            case ForecastType.DAILY:
                periods = 1
                freq = "D"
            case ForecastType.WEEKLY:
                periods = 7
                freq = "D"
            case ForecastType.MONTHLY:
                periods = NUM_DAYS_IN_MONTH
                freq = "D"
            case ForecastType.YEARLY:
                periods = NUM_DAYS_IN_YEAR
                freq = "D"
            case _:
                raise ValueError("Invalid forecast type specified")

        return X.shift(periods=periods, freq=freq)
