import sys
import os

sys.path.append("..")

import numpy as np
import pandas as pd
import pickle
import warnings
import re

from NHS_PROMs.settings import config
from NHS_PROMs.model import pl, param_grid
from NHS_PROMs.load_data import load_proms, structure_name
from NHS_PROMs.preprocess import filter_in_range, filter_in_labels
from NHS_PROMs.utils import (
    most_recent_file,
    downcast,
    map_labels,
    fillna_categories,
    pd_fit_resample,
    infer_categories_fit,
    KindSelector,
    get_feature_names,
    remove_categories,
)
from NHS_PROMs.data_dictionary import meta_dict

import shap

shap.initjs()

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import classification_report
from sklearn import set_config
from sklearn.utils.validation import check_is_fitted

set_config(display="diagram")

from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

# use adjusted fillna which can cope with non-existing categories for CategoricalDtype
pd.core.frame.DataFrame.fillna = fillna_categories
# added a remove categories
pd.core.frame.Series.remove_categories = remove_categories
# enable autodetect of categories from CategoricalDtype by using "infer" for SMOTENC
SMOTENC.fit_resample = pd_fit_resample(SMOTENC.fit_resample)
# enable inference of categories for encoders from CategoricalDtype
OneHotEncoder.fit = infer_categories_fit(OneHotEncoder.fit)
OrdinalEncoder.fit = infer_categories_fit(OrdinalEncoder.fit)


class PROMsModel():
    """
    Model handling class for PROMS models.
    """

    def __init__(self, kind="hip"):
        """
        Initiation for PROMsModel.

        Parameters
        ----------
        kind: str
            Determines what kind of surgery models are based on {"hip", "knee"}.
        """

        self.kind = kind
        self.outputs = config["outputs"][kind]

    def load_data(self, mode="train"):
        """
        Loads and preprocesses data based on the kind of surgery and if it is used for training or testing.

        Parameters
        ----------
        mode: str
            indicator to split data into a train or test set {"train", "test"}.

        Returns
        -------
        pd.DataFrame
            A DataFrame with all columns (before and after surgery).

        """
        df = (
            load_proms(part=self.kind)
                .apply(downcast)
                .rename(structure_name, axis=1)
        )

        self.load_meta(df.columns)

        df = self.preprocess(df)

        if mode == "train":
            df = df.query("t0_year != 'April 2019 - April 2020'").drop(columns="t0_year")
        elif mode == "test":
            df = df.query("t0_year == 'April 2019 - April 2020'").drop(columns="t0_year")
        else:
            raise ValueError(f"No valid mode: '{mode}'")

        return df

    def load_meta(self, columns):
        """
        Loads metadata based on column names into class attribute self.meta.

        Parameters
        ----------
        columns: (list, index)
            Iterable with the structured column names.

        Returns
        -------

        """
        # get meta data
        full_meta = {t + k: v for k, v in meta_dict.items() for t in ["t0_", "t1_"]}
        self.meta = {k: v for k, v in full_meta.items() if k in columns}

    def preprocess(self, df):
        """
        Preprocesses a DataFrame based on meta data and configuration.

        Parameters
        ----------
        df: pd.DataFrame
            Input DataFrame with columns as features and rows as samples.

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame to be used by PROMsModels.

        """
        # remove certain columns
        endings = config["preprocessing"]["remove_columns_ending_with"]
        cols2drop = [c for c in df.columns if c.endswith(endings)]

        df = (
            df.apply(lambda s: filter_in_range(s, **self.meta[s.name]))  # filter in range numeric features
                .apply(lambda s: filter_in_labels(s, **self.meta[
                s.name]))  # filter in labels categorical features + ordinal if ordered
                .apply(lambda s: map_labels(s, **self.meta[s.name]))  # map the labels as values for readability
                .query("t0_revision_flag == 'no revision'")  # drop revision cases
                .drop(columns=cols2drop)  # drop not needed columns
        )

        # remove low info values from columns (almost redundant) values
        for col, value in config["preprocessing"]["remove_low_info_categories"].items():
            df[col] = df[col].remove_categories(value)

        # remove NaNs/missing/unknown from numerical and ordinal features
        df = (
            df.apply(pd.Series.remove_categories, args=(["missing", "not known"],))
                .dropna(subset=KindSelector(kind="numerical")(df) + KindSelector(kind="ordinal")(df))
                .reset_index(drop=True)  # make index unique (prevent blow ups when joining)
        )

        return df

    def split_XY(self, df):
        """
        Splits a DataFrame into an feature set (X) and a label (Y) set.
        Y can have multiple columns (based on configuration).

        Parameters
        ----------
        df: pd.DataFrame
            A preprocessed input DataFrame.

        Returns
        -------
        tuple(pd.DataFrame, pd.DataFrame)
            The feature and label datasets as a tuple (X, Y).
        """

        # define inputs and outputs
        X = df.filter(regex="t0").copy()
        Y = df[self.outputs].copy()

        # get cut from settings
        for col in Y.columns:
            if pd.api.types.is_numeric_dtype(Y[col]):
                Y[col] = pd.cut(
                    Y[col],
                    include_lowest=True,
                    **self.outputs[col],
                )

        return X, Y

    def train_models(self):
        """
        Trains all models (for every output defined in configuration).

        Returns
        -------
        self

        """
        X, Y = (
            self.load_data(mode="train")
                .pipe(self.split_XY)
        )
        self.models = dict()
        for col, y in Y.iteritems():
            self.models[col] = self.train_model(X, y)
        return self

    def train_model(self, X, y):
        """
        Trains a individual model (defined in model.py) with a based in features and labels.

        Parameters
        ----------
        X: pd.DataFrame
            Feature set for the model.
        y: pd.Series
            Label set for the model.

        Returns
        -------
        sklearn estimator object
            A trained estimator object for the particular label set.
            Standard a GridSearchCV object is return, but depends of definition in model.py
        """

        GS = GridSearchCV(
            estimator=pl,
            param_grid=param_grid,
            scoring=config["score"]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            GS.fit(X, y)
        return GS

    def save_models(self, filename=None):
        """
        Saves models to a file. If no filename is given, the file is named in the format {kind}_{sha(5)}_.mld.
        The location is defined via the configuration.

        Parameters
        ----------
        filename: str
            Name of file to store the models in.

        Returns
        -------

        """

        if filename is None:
            hashable = frozenset(self.models.items())
            sha = hex(hash(hashable))[-5:]
            path = os.path.join("..", config["models"]["path"])
            filename = f"{self.kind}_{sha}.mdl"
        pickle.dump(self.models, open(os.path.join(path, filename), 'wb'))

    def load_models(self, filename=None):
        """
        Loads the saved models. If no filename is given, the last model for this PROMs kind is loaded.
        The location is defined via the configuration.

        Parameters
        ----------
        filename: str
            Name of file to load the models from.
        Returns
        -------
        self
        """

        path = os.path.join("..", config["models"]["path"])
        if filename is None:
            filename = most_recent_file(path, ext=".mdl", prefix=self.kind)
            if filename is None:
                raise ValueError("No correct models found!")
        else:
            if not re.search(fr"^{self.kind}_", filename):
                raise Warning(f"File '{filename} does not seem to be having models for {self.kind}")
        self.models = pickle.load(open(os.path.join(path, filename), 'rb'))
        return self

    def predict(self, X):
        """
        Makes predictions for all models.

        Parameters
        ----------
        X: pd.DataFrame
            The feature set to make predictions on.

        Returns
        -------
        dict()
            A dictionary with keys being the outputs and values being the predictions.

        """

        y_hat = dict()
        for name, model in self.models.items():
            check_is_fitted(model)
            y_hat[name] = model.predict(X)
        return y_hat

    def predict_proba(self, X):
        """
        Makes probability predictions for all models.

        Parameters
        ----------
        X: pd.DataFrame
            The feature set to make predictions on.

        Returns
        -------
        dict()
            A dictionary with keys being the outputs and values being the probability predictions.
        """

        y_hat = dict()
        for name, model in self.models.items():
            check_is_fitted(model)
            y_hat[name] = model.predict_proba(X)

            # reorder labels according to configuration
            org_labels = list(model.classes_)
            new_labels = config["outputs"][self.kind][name]["labels"]
            i = [org_labels.index(label) for label in new_labels]
            y_hat[name] = y_hat[name][:, i]

        return y_hat

    def classification_reports(self):
        """
        Prints a classification report for every model/output based on the test data.

        Returns
        -------

        """
        data = self.load_data(mode="test")
        X, Y = self.split_XY(data)
        for name, model in self.models.items():
            check_is_fitted(model)
            y_hat = model.predict(X)
            print(f"\nClassification report for {name}:\n")
            print(classification_report(Y[name], y_hat))

    def get_explainer(self, name):
        """
        Sets and gets the SHAP explainer (Tree) for a certain output.

        Parameters
        ----------
        name: str
            The output to get the explainer for.

        Returns
        -------
        shap.TreeExplainer
            The explainer.
        """

        if hasattr(self, "explainers") is False:
            self.explainers = dict()

        if self.explainers.get(name) is None:
            model = self.models[name]
            check_is_fitted(model)
            self.explainers[name] = shap.TreeExplainer(
                model.best_estimator_.named_steps["model"],
                #                 feature_perturbation='interventional',
                #                 model_output="probability",
                #                 data=self.load_data("train"),
            )
        return self.explainers[name]

    def force_plot(self, X, name):
        """
        Plots a SHAP force plot for a single prediction for a single output.

        Parameters
        ----------
        X: pd.DataFrame
            Feature set to make force plot for (length should be 1).
        name: str
            Output to make force plt for.

        Returns
        -------
        shap.force_plot
            Force plot object.
        """

        if X.shape[0] != 1:
            raise ValueError("First dimension should be 1. Expected a single case for force plot!")

        model = self.models[name]
        check_is_fitted(model)
        explainer = self.get_explainer(name)

        # rescaling base values for multiclass https://evgenypogorelov.com/multiclass-xgb-shap.html
        def logodds_to_proba(logodds):
            return np.exp(logodds) / np.exp(logodds).sum()

        predict_proba = model.predict_proba(X)
        i_max = np.argmax(predict_proba)
        end_value = predict_proba[0, i_max]
        X_pre = model.best_estimator_[:-1].transform(X)
        shap_values = explainer.shap_values(X_pre)[i_max]
        base_value = logodds_to_proba(explainer.expected_value)[i_max]
        feature_names = [re.sub("(t0_|gender_|_yes|_no|)", "", n).replace("_", " ") for n in get_feature_names(model)]
        out_names = f"{name} = {self.labels_encoded()[name][i_max]}"

        # rescaling according to https://github.com/slundberg/shap/issues/29
        shap_values = shap_values / shap_values.sum() * (end_value - base_value)

        fp = shap.force_plot(
            base_value=base_value,
            shap_values=shap_values,
            #             features=X_pre,
            feature_names=feature_names,
            out_names=out_names,
            #             link="logit",
        )
        return fp

    def force_plots(self, X=None):
        """
        Make SHAP force plots for all outputs.
        If no feature set X is given, a random sample from the test data is loaded.

        Parameters
        ----------
        X: pd.DataFrame
            Feature set to make force plot for (length should be 1).

        Returns
        -------

        """

        if X is None:
            df_data = self.load_data("test").sample()
            X, Y = self.split_XY(df_data)

        for name in self.models:
            display(self.force_plot(X, name))
