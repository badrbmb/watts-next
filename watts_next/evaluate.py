import numpy as np
import sklearn.metrics


def get_overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, np.float64]:
    """Get overall performance metrics."""
    overall_metrics = {
        "num_samples": np.float64(len(y_true)),
    }
    metrics = [
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_error",
    ]
    for metric_name in metrics:
        metric = getattr(sklearn.metrics, metric_name)
        overall_metrics[metric_name] = metric(y_true, y_pred)
    # Add RMSE
    overall_metrics["root_mean_squared_error"] = np.sqrt(
        overall_metrics["mean_squared_error"],
    )

    return overall_metrics
