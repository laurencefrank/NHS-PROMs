import pandas as pd

def downcast(s, try_numeric=True, category=False):
    """
    Downcasts a series to the lowest possible memory type

    Parameters
    ----------
    s
    try_numeric
    category

    Returns
    -------

    """

    if try_numeric:
        s = pd.to_numeric(s, errors="ignore")

    if category:
        if s.dtype.kind == "O":
            s = s.astype("category")

    if s.dtype.kind == "f":
        s = pd.to_numeric(s, errors="ignore", downcast="float")
    elif s.dtype.kind == "i":
        s = pd.to_numeric(s, errors="ignore", downcast="signed")
        s = pd.to_numeric(s, errors="ignore", downcast="unsigned")

    return s


def map_labels(series, kind="categorical", labels=None, backwards=False, **arg):
    """
    Maps a Series values by the labels given.

    Parameters
    ----------
    series: pd.Series
        Series to map on.
    kind: str
        Indicator for kind of data in series. With kind of {"categorical", "ordinal"}  the mapping is applied, otherwise not.
    labels: dict
        Defines with the mapping {key_0: value_0, etc.}.
    arg:
        Additional arguments.

    Returns
    -------
    pd.Series
        Series with mapped values.

    """

    if kind in ["categorical", "ordinal"]:
        if isinstance(labels, dict):
            if backwards:
                labels = {v: k for k, v in labels.items()}
            series = series.map(labels)
    return series
