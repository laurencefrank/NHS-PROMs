# import numpy as np
import pandas as pd
# import warnings
# import re
# import plotly.express as px
# import plotly.graph_objects as go

from NHS_PROMs.settings import config
# from NHS_PROMs.load_data import load_proms, structure_name
# from NHS_PROMs.preprocess import filter_in_range, filter_in_labels, method_delta
from NHS_PROMs.utils import (
    # downcast,
    # map_labels,
    fillna_categories,
    pd_fit_resample,
    infer_categories_fit,
    KindSelector,
    # get_feature_names,
    remove_categories,
)
# from NHS_PROMs.data_dictionary import meta_dict, methods

# import shap
# shap.initjs()

from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBRFClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.compose import make_column_selector

# from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import (
    ColumnTransformer,
    # make_column_transformer,
    # make_column_selector,
)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import (
#     RandomForestClassifier,
#     AdaBoostClassifier,
#     GradientBoostingRegressor,
#     BaggingClassifier,
# )
# from sklearn.metrics import classification_report, balanced_accuracy_score
# from sklearn.inspection import permutation_importance
from sklearn import set_config
set_config(display="diagram")

# from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier, EasyEnsembleClassifier
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline, make_pipeline
# from imblearn.under_sampling import RandomUnderSampler

# use adjusted fillna which can cope with non-existing categories for CategoricalDtype
pd.core.frame.DataFrame.fillna = fillna_categories
# added a remove categories
pd.core.frame.Series.remove_categories = remove_categories
# enable autodetect of categories from CategoricalDtype by using "infer" for SMOTENC
SMOTENC.fit_resample = pd_fit_resample(SMOTENC.fit_resample)
# enable inference of categories for encoders from CategoricalDtype
OneHotEncoder.fit = infer_categories_fit(OneHotEncoder.fit)
OrdinalEncoder.fit = infer_categories_fit(OrdinalEncoder.fit)

# use adjusted fillna which can cope with non-existing categories for CategoricalDtype
pd.core.frame.DataFrame.fillna = fillna_categories
# added a remove categories
pd.core.frame.Series.remove_categories = remove_categories
# enable autodetect of categories from CategoricalDtype by using "infer" for SMOTENC
SMOTENC.fit_resample = pd_fit_resample(SMOTENC.fit_resample)
# enable inference of categories for encoders from CategoricalDtype
OneHotEncoder.fit = infer_categories_fit(OneHotEncoder.fit)
OrdinalEncoder.fit = infer_categories_fit(OrdinalEncoder.fit)



# implemented automatic weights in pl
class BalancedXGBRFClassifier(XGBRFClassifier):

    def fit(self, X, y, **kwargs):
        weights = compute_sample_weight(class_weight="balanced", y=y)
        kwargs.update({"sample_weight": weights})

        return super().fit(X, y, **kwargs)

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

pl = Pipeline(
    (
        ("balancer", "passthrough"),
        ("by_column_kinds", ct),
        ("model", BalancedXGBRFClassifier()),
    )
)

param_grid = [
    {
        "model__n_estimators": [4] #[8*2**e for e in range(8)],
    }
]

# GS = GridSearchCV(
#     estimator=pl,
#     param_grid=param_grid,
#     scoring=config["score"]
# )



