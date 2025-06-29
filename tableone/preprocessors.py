import numpy as np
import pandas as pd

from .exceptions import InputError


def ensure_list(arg, arg_name):
    """
    Ensure input argument is a list.
    """
    if arg is None:
        return []
    elif isinstance(arg, str):
        return [arg]
    elif isinstance(arg, list):
        return arg
    else:
        raise TypeError(f"{arg_name} must be a string or a list of strings.")


def detect_categorical(data, groupby) -> list:
    """
    Detect categorical columns if they are not specified.

    Parameters
    ----------
        data : pandas DataFrame
            The input dataset.
        groupby : str (optional)
            The groupby variable.

    Returns
    ----------
        likely_cat : list
            List of variables that appear to be categorical.
    """
    # assume all non-numerical and date columns are categorical
    numeric_cols = set(data._get_numeric_data().columns.values)
    date_cols = set(data.select_dtypes(include=[np.datetime64]).columns)
    likely_cat = set(data.columns) - numeric_cols
    likely_cat = list(likely_cat - date_cols)

    # check proportion of unique values if numerical
    for var in data._get_numeric_data().columns:
        likely_flag = 1.0 * data[var].nunique()/data[var].count() < 0.005
        if likely_flag:
            likely_cat.append(var)

    if groupby:
        likely_cat = [x for x in likely_cat if x != groupby]

    return likely_cat


def order_categorical(data, order):
    """
    Define an order for categorical variables.
    """
    # if input df has ordered categorical variables, get the order.
    order_cats = [x for x in data.select_dtypes("category")
                  if data[x].dtype.ordered]  # type: ignore

    if any(order_cats):
        d_order_cats = {v: data[v].cat.categories for v in order_cats}
        d_order_cats = {k: ["{}".format(v) for v in d_order_cats[k]]
                        for k in d_order_cats}

    # combine the orders. custom order takes precedence.
    if order_cats and order:
        new = {**order, **d_order_cats}  # type: ignore
        for k in order:
            new[k] = order[k] + [x for x in new[k] if x not in order[k]]
        order = new
    elif order_cats:
        order = d_order_cats  # type: ignore

    return order


def get_groups(data, groupby, order, reserved_columns):
    """
    Get groups for table.

    If groupby is not specified, there will be a single "overall" group.
    """
    if groupby:
        groupbylvls = sorted(data.groupby(groupby).groups.keys())  # type: ignore

        # reorder the groupby levels if order is provided
        if order and groupby in order:
            unordered = [x for x in groupbylvls if x not in order[groupby]]
            groupbylvls = order[groupby] + unordered

        # check that the group levels do not include reserved words
        for level in groupbylvls:
            if level in reserved_columns:
                raise InputError("""Group level contains '{}', a reserved
                                    keyword.""".format(level))
    else:
        groupbylvls = ['Overall']

    return groupbylvls


def handle_categorical_nulls(df: pd.DataFrame, categorical: list, null_value: str = 'None') -> pd.DataFrame:
    """
    Replace missing values (<NA>, np.nan, pd.NA) in categorical columns with the specified null_value (e.g., 'Not Reported').
    Remove <NA> as a category if present, and ensure null_value is a category.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the categorical data.
    - categorical (list): List of categorical variables.
    - null_value (str): The string to replace null values with. Default is 'None'.

    Returns:
    - pd.DataFrame: The modified DataFrame.
    """
    df = df.copy()
    for column in categorical:
        # Replace missing values with null_value
        df[column] = df[column].astype('category')
        if null_value not in df[column].cat.categories:
            df[column] = df[column].cat.add_categories([null_value])
        df[column] = df[column].fillna(null_value)
        # Remove <NA> as a category if present (pandas 1.3+)
        if pd.NA in df[column].cat.categories:
            df[column] = df[column].cat.remove_categories([pd.NA])
    return df
