from typing import Any

import numpy as np
import pandas as pd
import sklearn.metrics
import structlog
from snorkel.slicing import PandasSFApplier, SlicingFunction, slicing_function

logger = structlog.get_logger()


def get_overall_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, Any]:
    """Get overall performance metrics."""
    mask = ~np.isnan(y_pred)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    if (count_non_na_samples := len(y_true_filtered)) < (count_samples := len(y_true)):
        logger.warning(
            event="Input y_pred contains NaNs. Computing metric on non-NaN subset only!",
            total_samples=count_samples,
            total_non_nan_samples=count_non_na_samples,
        )

    overall_metrics = {
        "num_samples": np.float64(count_samples),
        "num_non_nan_samples": np.float64(count_non_na_samples),
    }
    metrics = [
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_error",
    ]
    for metric_name in metrics:
        metric = getattr(sklearn.metrics, metric_name)
        overall_metrics[metric_name] = metric(y_true_filtered, y_pred_filtered)
    # Add RMSE
    overall_metrics["root_mean_squared_error"] = np.sqrt(
        overall_metrics["mean_squared_error"],
    )

    return overall_metrics


class SlicingFunctionsRepository:
    def __init__(self, min_cut_off: float = 0.1, max_cut_off: float = 0.9) -> None:
        self.min_cut_off = min_cut_off
        self.max_cut_off = max_cut_off

    @staticmethod
    def create_equality_slicing_function(field: str, value: ..., name: str) -> SlicingFunction:
        """Create an equality test slicing function."""

        @slicing_function()
        def slicing_fn(x: pd.Series) -> bool:
            return x[field] == value

        slicing_fn.name = name
        return slicing_fn

    @staticmethod
    def create_inequality_slicing_function(
        field: str,
        threshold: float,
        name: str,
        is_greater: bool,
    ) -> SlicingFunction:
        """Create an inequality test slicing function."""
        if is_greater:

            @slicing_function()
            def slicing_fn(x: pd.Series) -> bool:
                return x[field] >= threshold
        else:

            @slicing_function()
            def slicing_fn(x: pd.Series) -> bool:
                return x[field] <= threshold

        slicing_fn.name = name
        return slicing_fn

    def calculate_cut_offs(self, df_data: pd.DataFrame) -> dict[str, float]:
        """Compute all cut-offs based on quantiles."""
        cut_offs = {
            "value_high": df_data["value"].quantile(self.max_cut_off),
            "value_low": df_data["value"].quantile(self.min_cut_off),
            "t2m_cold": df_data["t2m"].quantile(self.min_cut_off),
            "t2m_hot": df_data["t2m"].quantile(self.max_cut_off),
            "tp_wet": df_data["tp"].quantile(self.max_cut_off),
            "windy": max(
                df_data["u10"].quantile(self.max_cut_off),
                df_data["v10"].quantile(self.max_cut_off),
            ),
        }
        return cut_offs

    def create_hourly_slicing_functions(self) -> list[SlicingFunction]:
        """Create slicing functions for each hour of the day."""
        hourly_slicing_functions = []
        for hour in range(24):
            hourly_slicing_fn = self.create_equality_slicing_function(
                field="hour_of_day",
                value=hour,
                name=f"is_hour_{hour}",
            )
            hourly_slicing_functions.append(hourly_slicing_fn)
        return hourly_slicing_functions

    def get_all_slicing_functions(self, df_data: pd.DataFrame) -> list[SlicingFunction]:
        """Create all slicing functions."""
        cut_offs = self.calculate_cut_offs(df_data)
        all_slicing_functions = [
            self.create_equality_slicing_function("is_holiday", 1, "is_holidays"),
            self.create_equality_slicing_function("is_weekend", 1, "is_weekend"),
            self.create_inequality_slicing_function(
                "value",
                cut_offs["value_high"],
                "high_value",
                is_greater=True,
            ),
            self.create_inequality_slicing_function(
                "value",
                cut_offs["value_low"],
                "low_value",
                is_greater=False,
            ),
            self.create_inequality_slicing_function(
                "t2m",
                cut_offs["t2m_cold"],
                "cold_temperature",
                is_greater=False,
            ),
            self.create_inequality_slicing_function(
                "t2m",
                cut_offs["t2m_hot"],
                "hot_temperature",
                is_greater=True,
            ),
            self.create_inequality_slicing_function(
                "tp",
                cut_offs["tp_wet"],
                "wet_precipitation",
                is_greater=True,
            ),
            self.create_inequality_slicing_function(
                "u10",
                cut_offs["windy"],
                "high_u10_wind",
                is_greater=True,
            ),
            self.create_inequality_slicing_function(
                "v10",
                cut_offs["windy"],
                "high_v10_wind",
                is_greater=True,
            ),
        ]
        # add hourly slicing functions
        all_slicing_functions.extend(self.create_hourly_slicing_functions())
        return all_slicing_functions

    @staticmethod
    def get_slices(all_slicing_functions: list[SlicingFunction], df: pd.DataFrame) -> np.ndarray:
        """Compute slices given a list of slicing functions and df."""
        # TODO: issue with snorkel typing SlicingFunction <> LabellingFunction
        return PandasSFApplier(all_slicing_functions).apply(df)  # type: ignore
