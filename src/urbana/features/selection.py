"""Feature selection."""
from typing import Callable

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


class KBestSelector:
    def __init__(
        self,
        score_func: Callable,
        preprocessor: Pipeline,
        power_transformer: PowerTransformer,
        y: pd.Series,
    ):
        """Perform K-Best selection.

        Args:
            score_func (callable): score function for sklearn SelectKBest
            preprocessor (Pipeline): sklearn preprocessor pipeline
            power_transformer (PowerTransformer): sklearn power transformer
            y (pd.Series): y data series
        """
        self.score_func = score_func
        self.preprocessor = preprocessor
        self.power_transformer = power_transformer
        self.y = y

    def k_best_selection(self, features: pd.DataFrame, k: int) -> pd.Index:
        """Select KBest based on sklearn.

        Args:
            features (pd.DataFrame): features
            k (int): number of features to select

        Returns:
            pd.Index: the k selected features
        """
        kbest_features = SelectKBest(score_func=self.score_func, k=k).fit(
            self.preprocessor.fit_transform(features),
            self.power_transformer.fit_transform(self.y.values.reshape(-1, 1)),
        )

        selected_features_cols = kbest_features.get_support(indices=True)
        return features.columns[selected_features_cols]
