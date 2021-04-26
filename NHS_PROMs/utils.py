import os
import pandas as pd
import operator
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
import imblearn
import sklearn

def most_recent_file(path, ext=None, prefix=None):
    if ext is None:
        ext=""
    if prefix is None:
        prefix = ""
    files = {f.name: os.stat(f).st_mtime for f in os.scandir(path) if f.name.endswith(ext) and f.name.startswith(prefix)}
    if files:
        latest_file = max(files.items(), key=operator.itemgetter(1))[0]
        return latest_file

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
                series = series.cat.add_categories(value)
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


def infer_categories_fit(func):
    """
    Decorator that enables Encoders with categories (like OneHotEncoder, OrdinalEncoder) to infer categories from
    catagories if dtype is CategoricalDtype by using the argument categories="categories".
    Parameters
    ----------
    func:
        Encoder's fit function (og OneHotEncoder.fit)

    Returns
    -------
    Decorated fit function that has the ability to process the argument categories="categories".

    """
    def inner_func(self, X, y=None):
        if self.categories == "categories" and (X.dtypes == "category").all():
            self.categories = [col.categories.to_list() for col in X.dtypes]

        return func(self, X, y)

    return inner_func


class KindSelector:
    """
    Selects on kind of dtype of column of DataFrame to select for eg. column trabsformer in pipeline.
    kinds kan be:
    - numerical (dtype number)
    - categorical (dtype category, not ordered)
    - ordinal (dtype category, ordered)
    """
    def __init__(self, kind):
        """
        Defines kind of selection on init
        Parameters
        ----------
        kind: str
            kind to select on {"numerical", "categorical", "ordinal"}.
        """
        self.kind = kind

    def __call__(self, df):
        if self.kind == "numerical":
            cols = df.select_dtypes("number").columns.to_list()
        elif self.kind == "categorical":
            cols = df.select_dtypes("category").columns
            cols = [col for col in cols if not df[col].cat.ordered]
        elif self.kind == "ordinal":
            cols = df.select_dtypes("category").columns
            cols = [col for col in cols if df[col].cat.ordered]
        return cols


def get_feature_names(sklobj, feature_names=None):
    """
    Extract feature names from pipeline with encoders

    Parameters
    ----------
    sklobj: sklearn pipeline or estimator
        pipeline to extract feature names from
    feature_names: list
        starting columns if no column transformer is present for example.

    Returns
    -------
    list
        list of feature names
    """
    if feature_names is None:
        feature_names = []

    if isinstance(sklobj, GridSearchCV):
        feature_names = get_feature_names(sklobj.best_estimator_, feature_names)
    elif isinstance(sklobj, (imblearn.pipeline.Pipeline, sklearn.pipeline.Pipeline)):
        for name, step in sklobj.steps:
            feature_names = get_feature_names(step, feature_names)
    elif isinstance(sklobj, ColumnTransformer):
        for name, transformer, columns in sklobj.transformers_:
            feature_names += get_feature_names(transformer, columns)
    elif isinstance(sklobj, OneHotEncoder):
        feature_names = sklobj.get_feature_names(feature_names).tolist()
    elif isinstance(sklobj, str):
        if sklobj == "passthrough":
            pass
        elif sklobj == "drop":
            feature_names = []

    return feature_names


def remove_categories(self, removals):
    """
    As pd.Series.cat.remove_categories() but checks if is category dtype and category exists.
    Parameters
    ----------
    self
    removals:
        categories to remove, just like pd.Series.remove_categories

    Returns
    -------
    Series with categories removed (values being NaN's now)
    """

    if hasattr(self, "cat"):
        if isinstance(removals, str):
            removals = [removals]
        removals = [cat for cat in removals if cat in self.cat.categories]
        if removals:
            self = self.cat.remove_categories(removals=removals)
    return self