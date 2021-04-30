import pandas as pd

from NHS_PROMs.utils import (
    fillna_categories,
    pd_fit_resample,
    infer_categories_fit,
    KindSelector,
    remove_categories,
)

from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBRFClassifier

from sklearn.compose import (
    ColumnTransformer,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from sklearn import set_config
set_config(display="diagram")

from imblearn.pipeline import Pipeline

# use adjusted fillna which can cope with non-existing categories for CategoricalDtype
# pd.core.frame.DataFrame.fillna = fillna_categories
# added a remove categories
# pd.core.frame.Series.remove_categories = remove_categories

# enable inference of categories for encoders from CategoricalDtype
OneHotEncoder.fit = infer_categories_fit(OneHotEncoder.fit)
OrdinalEncoder.fit = infer_categories_fit(OrdinalEncoder.fit)



class BalancedXGBRFClassifier(XGBRFClassifier):
    """
    Implementation of XGBRFClassifier with automatic weights handling for fit.
    """

    def fit(self, X, y, **kwargs):
        """
        Fit BalancedXGBRFClassifier.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary
        Returns
        -------
        self : returns an instance of self.
        """
        weights = compute_sample_weight(class_weight="balanced", y=y)
        kwargs.update({"sample_weight": weights})
        return super().fit(X, y, **kwargs)

# column transformer based on type of columns
ct = ColumnTransformer(
    (
        ("numerical", StandardScaler(), KindSelector(kind="numerical")),
        (
            "categorical",
            OneHotEncoder(categories="categories", handle_unknown="ignore"),
            KindSelector(kind="categorical"),
        ),
        (
            "ordinal",
            OrdinalEncoder(categories="categories", handle_unknown="ignore"),
            KindSelector(kind="ordinal"),
        ),
    ),
    remainder="drop",
)

# kills warnings ;)
std_xgb_args = dict(
    use_label_encoder=True,
    objective="multi:softprob",
    eval_metric="mlogloss",
)

# standard pipeline for model training
pl = Pipeline(
    (
        ("balancer", "passthrough"),
        ("by_column_kinds", ct),
        ("model", BalancedXGBRFClassifier(**std_xgb_args)),
    )
)

# parameter grid for GridSearchCV
param_grid = [
    {
        "model__n_estimators": [8*2**e for e in range(7)],
    },
]



