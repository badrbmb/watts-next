import pandas as pd
from xgboost import XGBRegressor


class XGBoostRegressor(XGBRegressor):
    """Wrapper around XGBRegressor while specify some parameters in __init__ .

    Following scikit-learn estimators conventions.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        colsample_bytree: float = 0.8,
        **kwargs: ...,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            colsample_bytree=colsample_bytree,
            **kwargs,
        )

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Thin wrapper around parent class predict to format outputs."""
        y_pred = super().predict(X)
        return pd.Series(y_pred, index=X.index)
