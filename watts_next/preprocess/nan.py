import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NanPreProcessor(BaseEstimator, TransformerMixin):
    """Custom preprocessor class filling missing values in original features."""

    def __init__(
        self,
        fill_method: str = "linear",
        ts_freq: str = "H",
        fill_limit: int = 6,
        limit_area: str = "inside",
    ) -> None:
        self.ts_freq = ts_freq
        self.fill_method = fill_method
        self.fill_limit = fill_limit
        self.limit_area = limit_area

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "NanPreProcessor":  # noqa: ARG002
        """Nothing to fit."""
        return self

    def validate_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate the transformed df."""
        if (inferred_freq := pd.infer_freq(df.index)) != self.ts_freq:
            raise ValueError(
                f"Inferred freq. {inferred_freq} does not conform to target freq {self.ts_freq}",
            )
        df.index.freq = inferred_freq
        return df

    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:  # noqa: ARG002
        """Transform data."""
        df_data = X.copy()
        # resample to target ts_freq
        df_data = df_data.resample(self.ts_freq).first()

        # fill missing feature values only
        # for instance 3h NWP frequency vs hourly electricty data
        df_data = df_data.interpolate(
            method=self.fill_method,
            limit=self.fill_limit,
            limit_area=self.limit_area,
        )

        # make sure no more nans
        df_data = df_data.dropna()

        # validate and return
        return self.validate_df(df_data)
