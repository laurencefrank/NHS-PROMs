import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from NHS_PROMs.data_dictionary import methods

def clean_data(df_data: pd.DataFrame, df_meta: pd.DataFrame, replace_value=np.nan) -> pd.DataFrame:
    """
    Cleans dat based on meta data.

    Parameters
    ----------
    df_data : pd.DataFrame
        DataFrame to clean
    df_meta : pd.DataFrame
        Dataframe with e.g.range labels
    replace_value :
        The value to replace non conforming values with

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame

    """

    df = df_data.copy()
    for col in df.columns:
        if col in df_meta.index:
            labels = df_meta.loc[col, "labels"]
            range = df_meta.loc[col, "range"]
            if labels == labels:
                labels = [k for k, v in labels[0].items() if v != "missing"]
                df.loc[df[col].isin(labels) == False, col] = replace_value
            elif range == range:
                df.loc[df[col].between(*range) == False, col] = replace_value
    return df


def filter_in_range(series, kind="numerical", range=None, **arg):
    """
    Filters a Series values by the range they should be in.

    Parameters
    ----------
    series: pd.Series
        Series to filter on.
    kind: str
        Indicator for kind of data in series. With kind = "numerical" the filter is applied, otherwise not.
    range: tuple
        Defines the range values should be in (start, end).
    arg:
        Additional arguments.

    Returns
    -------
    pd.Series
        Series with filtered values.

    """

    if kind == "numerical":
        if isinstance(range, tuple):
            series[series.between(*range) == False] = np.nan
    return series


def filter_in_labels(series, kind="categorical", labels=None, **arg):
    """
    Filters a Series values by the labels they should be in by the use of a categorical dtype.

    Parameters
    ----------
    series: pd.Series
        Series to filter on.
    kind: str
        Indicator for kind of data in series. With kind of {"categorical", "ordinal"}  the filter is applied, otherwise not.
        With kind = "ordinal" the category dtype is considered to be ordered.
    labels: dict
        Defines with the keys the labels values should be in {key_0: value_0, etc.}.
    arg:
        Additional arguments.

    Returns
    -------
    pd.Series
        Series with filtered values.

    """

    if kind in ["categorical", "ordinal"]:
        if isinstance(labels, dict):
            ordered = kind == "ordinal"
            cat_type = CategoricalDtype(categories=labels.keys(), ordered=ordered)
            series = series.astype(cat_type)
    return series


def method_delta(df):
    """
    Extracts delta Dataframe from method features available on t0 and t1, where delta is defined as t1-t0

    Parameters
    ----------
    df: pd.DataFrame
        Input dataframe with columns structured like tx_method_feature.

    Returns
    -------
    pd.DataFrame
        Delta DataFrame with columns structured like delta_method_feature.

    """

    # create MultiIndex
    df = df.copy().sort_index(axis=1)
    df.columns = pd.MultiIndex.from_frame(
        df.columns.str.extract(fr"^(t[01])_({'|'.join(methods.keys())})?_?(.*)$"),
        names=["available", "method", "feature"],
    )
    # select only methods dim and scores + get delta (t1 - t0)
    df = df.loc[
        :, [(m == m) & (f not in ["profile", "predicted"]) for t, m, f in df.columns]
    ]
    df_delta = df["t1"] - df["t0"]

    df_delta.columns = ["delta_" + "_".join(col) for col in df_delta.columns]
    return df_delta