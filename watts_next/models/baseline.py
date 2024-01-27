from enum import Enum
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

pd.offsets.Day()

NUM_DAYS_IN_MONTH = 30
NUM_DAYS_IN_YEAR = 365


class ForecastType(Enum):
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class AheadHourlyForecastRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, forecast_type: ForecastType | str) -> None:
        if isinstance(forecast_type, str):
            forecast_type = ForecastType(forecast_type)
        self.forecast_type = forecast_type
        self._shifted_values = None

    @property
    def shifted_values(self) -> pd.Series | None:
        """Store for shifted values (post-fit)."""
        return self._shifted_values

    @shifted_values.setter
    def shifted_values(self, value: pd.Series) -> None:
        self._shifted_values = value

    def fit(
        self,
        X: Any,  # noqa: ANN401, ARG002
        y: pd.Series,
    ) -> "AheadHourlyForecastRegressor":
        """Shift the dataframe with the desired frequency and store shifted results."""
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
        self.shifted_values = y.shift(periods=periods, freq=freq)
        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Check fitted status and return a Boolean value."""
        return self._shifted_values is not None

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict based on forcasting type."""
        check_is_fitted(self)
        if self.shifted_values is None:
            # FIXME: convered by the _is_fitted (can remove this check)
            raise ValueError("Regressor must be fitted before calling predict.")

        # Align the predicted values with the requested timestamps
        return self.shifted_values.reindex(X.index)
