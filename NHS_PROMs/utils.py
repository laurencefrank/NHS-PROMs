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


def fillna_categories(self, value):
    """
    As pd.Series.fillna() or pd.DataFrame.fillna(), but adds a category first id dtype is category.
    Parameters
    ----------
    self
    value:
        Fill value, just like fillna()

    Returns
    -------
    NaN-less pd object
    """

    def fill_series(series):
        if series.isna().sum():
            if hasattr(series, "cat") and value not in series.cat.categories:
                series.cat.add_categories(value, inplace=True)
            series = series.fillna(value)
        return series

    if isinstance(self, pd.Series):
        return self.fill_series(value)
    elif isinstance(self, pd.DataFrame):
        return self.apply(fill_series)


def pd_fit_resample(func):
    """
    Decorator that enables imblearn SMOTENC to be used with on a pd.DataFrame with argument categorical_features being:
    - "infer" which determines categorical columns if the are of the dtype "category".
    - an iterable with the column names.

    Parameters
    ----------
    func: func
        function to wrap: SMOTENC._fit_resample

    Returns
    wrapped function
    -------

    """
    def inner_func(self, X, y):

        if (
            isinstance(self.categorical_features, str)
            and self.categorical_features == "infer"
            and isinstance(X, pd.DataFrame)
        ):
            self.categorical_features = X.dtypes == "category"
        elif (
            hasattr(self.categorical_features, "__iter__")
            and not isinstance(self.categorical_features, str)
            and isinstance(self.categorical_features[0], str)
            and isinstance(X, pd.DataFrame)
        ):
            self.categorical_features = [
                X.columns.get_loc(col) for col in self.categorical_features
            ]

        return func(self, X, y)

    return inner_func