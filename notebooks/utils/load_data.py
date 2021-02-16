import numpy as np
import pandas as pd
import os
from os import path
import requests
from zipfile import ZipFile
from io import BytesIO
import re
import warnings

from utils.data_dictionary import methods, column_meta

# from https://digital.nhs.uk/data-and-information/data-tools-and-services/data-services/patient-reported-outcome-measures-proms
URL_PROMS_DATA = [
    r"https://files.digital.nhs.uk/6C/A1D581/CSV%20Data%20Pack%202016-17%20Finalised.zip",
    r"https://files.digital.nhs.uk/70/5176AA/CSV%20Data%20Pack%20Final%201718.zip",
    r"https://files.digital.nhs.uk/52/A8FF7F/PROMs%20CSV%20Data%20Pack%20Finalised%202018-19.zip",
    r"https://files.digital.nhs.uk/1F/51FEDE/PROMs%20CSV%20Data%20Pack%20Provisional%201920.zip",
]

def load_proms(part, org="provider", data_path="../data"):

    # define path location
    file_name = f"{part}-{org}.parquet"
    full_path = path.join(data_path, file_name)

    # load from disk if present,
    # otherwise get it directly from NHS-source + rename columns + save to parquet
    if path.isfile(full_path):
        df_raw = pd.read_parquet(full_path)
    else:
        df_raw = read_online_proms_data(urls=URL_PROMS_DATA, part=part, org=org).apply(downcast)
        df_raw.columns = (
            df_raw.columns.str.replace("Pre-Op Q", "t0")
                .str.replace("Post-Op Q", "t1")
                .str.replace("Knee Replacement", "oks")
                .str.replace("Hip Replacement", "ohs")
                .str.replace("-", "_")
                .str.replace(" ", "_")
                .str.lower()
        )
        if len(df_raw)==0:
            print("No data found!")
        else:
            try:
                df_raw.to_parquet(full_path)
            except:
                print(f"Could not save {full_path}, but has the dataframe in memory.")

    return df_raw

def read_online_proms_data(urls, part="hip", org="provider"):
    df = pd.DataFrame()
    for url in urls:
        response = requests.get(url)
        with ZipFile(BytesIO(response.content)) as zipfile:
            p = re.compile(fr"^{part} replacements? {org} [\d]{{4}}.csv$", flags=re.I)
            zippedfiles = [file for file in zipfile.namelist() if p.match(file)]
            for zippedfile in zippedfiles:
                with zipfile.open(zippedfile) as thefile:
                    df_file = pd.read_csv(thefile, na_values=["*"]).apply(downcast)
                    print(f"loaded {zippedfile} from {url}.")
                df = pd.concat([df, df_file])
    return df.apply(downcast)

def downcast(s, try_numeric=True, category=True):
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


# source: https://techoverflow.net/2018/01/16/downloading-reading-a-zip-file-in-memory-using-python/
def download_extract_zip(url):
    """
    Download a ZIP file and extract its contents in memory
    yields (filename, file-like object) pairs
    """
    response = requests.get(url)
    with ZipFile(BytesIO(response.content)) as thezip:
        for zipinfo in thezip.infolist():
            with thezip.open(zipinfo) as thefile:
                yield zipinfo.filename, thefile


def dir_digger(path: str, **args) -> list:
    """
    Recursively looks for files in path.
    :param path: Root path to root through.
    :param args: E.g. filter = ".txt" (or tuple of strings) to filter filenames on.
    :return: List of file paths (str)
    """
    files = args.get("files", [])
    ext = args.get("ext", "")
    for sub_path in os.listdir(path):
        new_path = os.path.join(path, sub_path)
        if os.path.isfile(new_path) & new_path.endswith(ext):
            files.append(new_path)
        elif os.path.isdir(new_path):
            files = dir_digger(new_path, files=files, ext=ext)
    return files


def read_csv_zip(path: str, **args) -> pd.DataFrame:
    """
    Reads csv or zipped csv file and returns pandas DataFrame.
    :param path: Path to (zipped) csv file
    :param args: Additional arguments for pandas.read_csv()
    :return: The DataFrame
    """
    if path.endswith(".csv"):
        return pd.read_csv(path, **args)
    elif path.endswith(".csv.zip"):
        with ZipFile(path) as myzip:
            filename = os.path.splitext((os.path.basename(path)))[0]
            return pd.read_csv(myzip.open(filename), **args)


def structure_name(col: str) -> str:
    """
    Structures name of columns to tx_method_dim|other.
    :param col: Original name of col
    :return: New structured name
    """

    # fix misspelled sucess -> success + eq_vas -> eqvas
    col = re.sub("sucess", "success", col)
    col = re.sub(r"eq_vas", r"eqvas", col)
    col = re.sub(r"eq_5d", r"eq5d", col)

    # fix double prefix
    col = re.sub(fr"^((?:oks|ohs)_)(?={'|'.join(methods.keys())})", "", col)

    # fix eqvas add score + order
    col = re.sub(r"^(t[01]_eqvas)$", r"\1_score", col)
    col = re.sub(r"^(eqvas)_(t[01])(.*)", r"\2_\1\3", col)

    # fix eqd5 remove _index + add score + order
    col = re.sub(r"_index", "", col)
    col = re.sub(r"^(t[01]_eq5d)$", r"\1_score", col)
    col = re.sub(r"^(eq5d)_(t[01])(.*)", r"\2_\1\3", col)
    # retrieve eq5d dimensions
    col = re.sub(
        fr"(t[01])_({'|'.join(methods['eq5d']['dims']['names'])})", r"\1_eq5d_\2", col
    )

    # fix ohs + ohk order
    col = re.sub(r"^(ohs|oks)_(t[01])(.*)", r"\2_\1\3", col)

    # add_prefix to general info
    col = re.sub(r"^((?!t[01_]).*)$", r"t0_\1", col)

    return col


def get_meta(columns: list) -> pd.DataFrame:
    """
    Get meta DataFrame based on column names.
    :param columns: Index or list of strings with column names
    :return: DataFrame with meta data
    """
    meta = ["t", "method", "feature", "kind", "labels", "range"]
    df_meta = pd.DataFrame(index=columns, columns=meta)

    for column in columns:
        result = re.search(
            fr"^t(?P<t>[01])_?(?P<method>{'|'.join(methods)})?_(?P<feature>.*)$", column
        )
        if result is None:
            warnings.warn(f"Incorrect format for ({column}).")
        else:
            df_meta.loc[column, "feature"] = feature = result.group("feature")
            df_meta.loc[column, "t"] = t = result.group("t")
            method = result.group("method")
            if method:
                df_meta.loc[column, "method"] = method
                meta = column_meta.get(method + "_" + feature)
            else:
                meta = column_meta.get(feature)

            if meta:
                for key in ["kind", "labels", "range"]:
                    value = meta.get(key, np.nan)
                    if isinstance(value, dict):
                        value = [value]
                    df_meta.loc[column, key] = value
            else:
                warnings.warn(f"Meta data was not found for {method} + {feature}.")

    return df_meta


def clean_data(df_data: pd.DataFrame, df_meta: pd.DataFrame, replace_value=np.nan) -> pd.DataFrame:
    """
    cleans dat based on meta data.
    :param df_data: DataFrame to clean
    :param df_meta: Dataframe with e.g.range labels
    :param replace_value: The value to replace non conforming values with
    :return: Cleaned DataFrame
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