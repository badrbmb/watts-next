from typing import Callable

import numpy as np
import pandas as pd
import sklearn.metrics
import structlog
from snorkel.slicing import PandasSFApplier, SlicingFunction, slicing_function

logger = structlog.get_logger()


def get_overall_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, np.float64]:
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
    def __init__(
        self,
        min_cut_off: float = 0.1,
        max_cut_off: float = 0.9,
    ) -> None:
        self.min_cut_off = min_cut_off
        self.max_cut_off = max_cut_off

    @staticmethod
    def is_cold(x: pd.Series, cut_off: float) -> bool:
        """Slicing function for cold timestamps."""
        return x["t2m"] <= cut_off

    @staticmethod
    def is_hot(x: pd.Series, cut_off: float) -> bool:
        """Slicing function for hot timestamps."""
        return x["t2m"] >= cut_off

    @staticmethod
    def is_windy(x: pd.Series, cut_off: float) -> bool:
        """Slicing function for windy timestamps."""
        return abs(x["u10"]) >= cut_off or abs(x["v10"]) >= cut_off

    @staticmethod
    def is_wet(x: pd.Series, cut_off: float) -> bool:
        """Slicing function for wet timestamps."""
        return x["tp"] >= cut_off

    @slicing_function()
    @staticmethod
    def is_holidays(x: pd.Series) -> bool:
        """Slicing function for holiday timestamps."""
        return x["holiday"] != "NA"

    @staticmethod
    def make_cut_off_slicing_function(function: Callable, cutoff: float) -> SlicingFunction:
        """Create the slicing functions with their mathcing cut-offs."""
        return SlicingFunction(
            name=function.__name__,
            f=function,
            resources={"cut_off": cutoff},
        )

    def get_all_slicing_function(self, df_data: pd.DataFrame) -> list[SlicingFunction]:
        """Dynamically define cut-off based slicing functions."""
        cut_offs = {}
        cut_offs["is_cold"] = df_data["t2m"].quantile(self.min_cut_off)
        cut_offs["is_hot"] = df_data["t2m"].quantile(self.max_cut_off)
        cut_offs["is_wet"] = df_data["tp"].quantile(self.max_cut_off)
        cut_offs["is_windy"] = max(
            [
                df_data["u10"].quantile(self.max_cut_off),
                df_data["v10"].quantile(self.max_cut_off),
            ],
        )

        all_slicing_functions = [self.is_holidays]
        for function in [
            self.is_cold,
            self.is_hot,
            self.is_wet,
            self.is_windy,
        ]:
            all_slicing_functions.append(
                self.make_cut_off_slicing_function(
                    function=function,
                    cutoff=cut_offs[function.__name__],
                ),
            )
        return all_slicing_functions

    @staticmethod
    def get_slices(all_slicing_functions: list[SlicingFunction], df: pd.DataFrame) -> np.ndarray:
        """Comptue slices given a list of slicing functions and df."""
        # TODO: issue with snorkel typing SlicingFunction <> LabellingFunction
        return PandasSFApplier(all_slicing_functions).apply(df)  # type: ignore
