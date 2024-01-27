import datetime as dt
from typing import Literal

import pandas as pd
from pandas.core.indexes.datetimes import DatetimeIndex
from prophet import Prophet

ProphetMode = Literal["default", "regressor"]
RegressorVariable = Literal["t2m", "tp", "u10", "v10"]


class ProphetPredictor:
    def __init__(
        self,
        mode: ProphetMode,
        country_iso2: str,
        seasonality_mode: Literal["additive", "multiplicative"] = "additive",
        holidays_prior_scale: float = 10,
        changepoint_prior_scale: float = 0.05,
        mcmc_samples: int = 0,
        regressor_variables: list[RegressorVariable] | None = None,
    ) -> None:
        self.mode = mode
        self.seasonality_mode = seasonality_mode
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_prior_scale = changepoint_prior_scale
        self.mcmc_samples = mcmc_samples
        self.regressor_variables = regressor_variables
        match self.mode:
            case "default":
                self.model_ = self._load_default_model()
            case "regressor":
                self.model_ = self._load_regressor_model()
            case _:
                raise ValueError(f"mode={self.mode} not supported!")
        self.model_.add_country_holidays(country_name=country_iso2)
        # variable to store df timezone
        self._time_zone = None

    @property
    def time_zone(self) -> dt.timezone | None:
        """Time zone of the fitted data."""
        return self._time_zone

    @time_zone.setter
    def time_zone(self, value: dt.timezone) -> None:
        self._time_zone = value

    def _load_default_model(self) -> Prophet:
        """Load a default model with defautl params."""
        return Prophet(
            seasonality_mode=self.seasonality_mode,
            holidays_prior_scale=self.holidays_prior_scale,
            changepoint_prior_scale=self.changepoint_prior_scale,
            mcmc_samples=self.mcmc_samples,
        )

    def _load_regressor_model(self) -> Prophet:
        """Load a regressor model."""
        model = self._load_default_model()
        if self.regressor_variables is None:
            raise ValueError(
                f"Need to specify `regressor_variables` when mode={self.mode}",
            )
        for regressor in self.regressor_variables:
            model.add_regressor(
                name=regressor,
            )
        return model

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProphetPredictor":
        """Fit the model."""
        if not isinstance(X.index, DatetimeIndex):
            raise ValueError(f"Expected DatetimeIndex, got {type(X.index)} instead.")
        # save the time zone
        self.time_zone = X.index.tzinfo
        df = (
            X.join(y)
            .tz_convert(None)
            .reset_index()
            .rename(
                columns={"value": "y", "timestamp": "ds"},
            )
        )
        self.model_.fit(df=df)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Return model predictions."""
        df = X.tz_convert(None).reset_index().rename(columns={"timestamp": "ds"})
        y_pred = self.model_.predict(df=df, vectorized=True)
        y_pred = y_pred.set_index("ds").rename(columns={"yhat": "value"})
        y_pred.index = y_pred.index.tz_localize(self.time_zone)
        return y_pred["value"]
