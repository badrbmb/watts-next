import numpy as np
import sklearn.metrics
import structlog

logger = structlog.get_logger()


def get_overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
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
