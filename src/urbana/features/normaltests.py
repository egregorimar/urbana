"""Normality tests.

Ref:
    https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
"""


import numpy as np
import pandas as pd
from numpy.random import seed
from scipy.stats import normaltest, shapiro

from urbana.constants import RANDOM_STATE

seed(RANDOM_STATE)


def normaltest_dagostino(data, alpha=0.05, verbose=False, n_threshold=20, **kwargs):
    # kurtosistest only valid for n>=20
    if len(data.dropna()) <= n_threshold:
        return np.nan

    # normality test
    stat, p = normaltest(data, **kwargs)

    if verbose is True:
        print("Statistics=%.3f, p=%.3f" % (stat, p))
    # interpret
    if p > alpha:
        if verbose is True:
            print("Sample looks Gaussian (fail to reject H0)")
        return True
    else:
        if verbose is True:
            print("Sample does not look Gaussian (reject H0)")
        return False


def normaltest_shapiro(data, alpha=0.05, n_threshold=3, verbose=False):
    if len(data.dropna()) <= n_threshold:
        return np.nan

    # normality test
    stat, p = shapiro(data.dropna())
    if verbose is True:
        print("Statistics=%.3f, p=%.3f" % (stat, p))
    if p > alpha:
        if verbose is True:
            print("Sample looks Gaussian (fail to reject H0)")
        return True
    else:
        if verbose is True:
            print("Sample does not look Gaussian (reject H0)")
        return False


def get_normaltest_df(
    df, alpha=0.05, shapiro_n_threshold=3, dagostino_n_threshold=20, verbose=False
):
    """Returns df with Normality test per index."""
    results_series = {}
    for measure, data in df.iterrows():
        result_dagostino = normaltest_dagostino(
            data,
            alpha=alpha,
            n_threshold=dagostino_n_threshold,
            nan_policy="omit",
            verbose=verbose,
        )
        result_shapiro = normaltest_shapiro(
            data, n_threshold=shapiro_n_threshold, alpha=alpha, verbose=verbose
        )
        results_series.update(
            {measure: {"dagostino": result_dagostino, "shapiro": result_shapiro}}
        )
    return pd.DataFrame(results_series).T
