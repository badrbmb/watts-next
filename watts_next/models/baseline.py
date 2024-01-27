from typing import Literal

import pandas as pd
from pandas.tseries.frequencies import to_offset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.compose import TransformedTargetForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.series.detrend import Deseasonalizer, Detrender

# Some approximations...
NUM_DAYS_IN_MONTH = 30
NUM_DAYS_IN_YEAR = 365

Frequency = Literal["H", "D", "W", "m", "Q", "Y"]


class TimeShiftForecastRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        freq: Frequency,
    ) -> None:
        self.freq = freq
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
        X: ...,  # noqa: ARG002
        y: pd.Series,
    ) -> "TimeShiftForecastRegressor":
        """Shift the dataframe with the desired frequency and store shifted results."""
        # TODO: refact this a bit awkward
        match self.freq:
            case "H":
                periods = 1
                _freq = "h"
            case "D":
                periods = 1
                _freq = "D"
            case "W":
                periods = 7
                _freq = "D"
            case "m":
                periods = NUM_DAYS_IN_MONTH
                _freq = "D"
            case "Q":
                periods = 3 * NUM_DAYS_IN_MONTH
                _freq = "D"
            case "Y":
                periods = NUM_DAYS_IN_YEAR
                _freq = "D"
            case _:
                raise ValueError(f"Invalid freq={self.freq} specified")
        self.shifted_values = y.shift(periods=periods, freq=_freq)
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


class SeasonalTrendDecompositionForecaster(BaseForecaster):
    def __init__(
        self,
        ref_freq: Frequency,
        seasonal_model: Literal["additive", "multiplicative"],
        polynomial_degree: int = 1,
        deseasonalize_daily: bool = True,
        deseasonalize_weekly: bool = True,
        deseasonalize_monthly: bool = True,
        deseasonalize_quarterly: bool = True,
    ) -> None:
        super().__init__()
        self.seasonal_model = seasonal_model
        self.ref_freq = ref_freq
        self.polynomial_degree = polynomial_degree
        self.deseasonalize_daily = deseasonalize_daily
        self.deseasonalize_weekly = deseasonalize_weekly
        self.deseasonalize_monthly = deseasonalize_monthly
        self.deseasonalize_quarterly = deseasonalize_quarterly

        self.model_ = TransformedTargetForecaster(
            steps=self._get_transformer_steps(),
        )

    @property
    def ref_freq_total_seconds(self) -> float:
        """Convert the reference frequency to total seconds."""
        return to_offset(self.ref_freq).delta.total_seconds()

    def _get_transformer_steps(self) -> list[tuple[str, ...]]:
        steps = []
        if self.deseasonalize_daily:
            steps.append(
                (
                    "deseasonalize_daily",
                    Deseasonalizer(
                        model=self.seasonal_model,
                        sp=self._get_seasonal_periodicity(target_freq="D"),
                    ),
                ),
            )
        if self.deseasonalize_weekly:
            steps.append(
                (
                    "deseasonalize_weekly",
                    Deseasonalizer(
                        model=self.seasonal_model,
                        sp=self._get_seasonal_periodicity(target_freq="W"),
                    ),
                ),
            )
        if self.deseasonalize_monthly:
            steps.append(
                (
                    "deseasonalize_monthly",
                    Deseasonalizer(
                        model=self.seasonal_model,
                        sp=self._get_seasonal_periodicity(target_freq="m"),
                    ),
                ),
            )
        if self.deseasonalize_quarterly:
            steps.append(
                (
                    "deseasonalize_quarterly",
                    Deseasonalizer(
                        model=self.seasonal_model,
                        sp=self._get_seasonal_periodicity(target_freq="Q"),
                    ),
                ),
            )

        # add detrenders
        steps.append(
            (
                "detrend",
                Detrender(forecaster=PolynomialTrendForecaster(degree=self.polynomial_degree)),
            ),
        )
        # add forecast
        steps.append(
            ("poly_forecast", PolynomialTrendForecaster(degree=self.polynomial_degree)),
        )

        return steps

    def _get_seasonal_periodicity(self, target_freq: Frequency) -> int:
        """Helper function to return the sp based on the ref_req."""
        match target_freq:
            case "D":
                return int(
                    pd.Timedelta(days=1).total_seconds() / self.ref_freq_total_seconds,
                )
            case "W":
                return 7 * self._get_seasonal_periodicity(target_freq="D")
            case "m":
                return NUM_DAYS_IN_MONTH * self._get_seasonal_periodicity(target_freq="D")
            case "Q":
                return 3 * self._get_seasonal_periodicity(target_freq="m")
            case "Y":
                return NUM_DAYS_IN_YEAR * self._get_seasonal_periodicity(target_freq="D")
            case _:
                raise ValueError(f"{target_freq=} is not valid")

    def fit(self, X: ..., y: pd.Series) -> "SeasonalTrendDecompositionForecaster":  # noqa: ARG002
        """Fit the model."""
        self.model_.fit(y, fh=y.index)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return prediction."""
        return self.model_.predict(fh=X.index)
