import warnings

import numpy as np
import pandas as pd


def docstring_copier(*sub):
    """
    Wrap the TableOne docstring (not ideal :/)
    """
    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*sub)
        return obj
    return dec


def set_display_options(max_rows=None,
                        max_columns=None,
                        width=None,
                        max_colwidth=None):
    """
    Set pandas display options. Display all rows and columns by default.
    """
    display_options = {'display.max_rows': max_rows,
                       'display.max_columns': max_columns,
                       'display.width': width,
                       'display.max_colwidth': max_colwidth}

    for k in display_options:
        try:
            pd.set_option(k, display_options[k])
        except ValueError:
            msg = """Newer version of Pandas required to set the '{}'
                        option.""".format(k)
            warnings.warn(msg)


def format_pvalues(table, pval, pval_adjust, pval_threshold, pval_digits):
    """
    Formats p-values to a fixed number of decimal places and optionally adds
    significance markers based on a threshold.
    """
    def _format(p):
        if pd.isnull(p):
            return ""
        try:
            fval = float(p)
        except ValueError:
            return str(p)
        if fval < 10**(-pval_digits):
            return f"<{10**(-pval_digits):.{pval_digits}f}"
        return f"{fval:.{pval_digits}f}"

    def _apply_format_and_threshold(col_name):
        if col_name in table.columns:
            numeric_mask = pd.to_numeric(table[col_name], errors='coerce').notna()
            asterisk_mask = numeric_mask & (table[col_name].astype(float) < pval_threshold)
            table[col_name] = table[col_name].apply(_format).astype(str)
            table.loc[asterisk_mask, col_name] += "*"

    if pval_adjust:
        col = 'P-Value (adjusted)'
        if pval_threshold:
            _apply_format_and_threshold(col)
        elif col in table.columns:
            table[col] = table[col].apply(_format).astype(str)

    elif pval:
        col = 'P-Value'
        if pval_threshold:
            _apply_format_and_threshold(col)
        elif col in table.columns:
            table[col] = table[col].apply(_format).astype(str)

    return table


def format_smd_columns(table, smd, smd_table):
    """
    Formats the SMD (Standardized Mean Differences) columns. Rounds the SMD values
    and ensures they are presented as strings.
    """
    if smd and smd_table is not None:
        for c in list(smd_table.columns):
            if c in table.columns:
                table[c] = table[c].apply(
                    lambda v: '{:.3f}'.format(v) if pd.api.types.is_numeric_dtype(type(v)) else 'N/A'
                ).astype(str)
                table.loc[table[c] == '0.000', c] = '<0.001'

    return table


def apply_limits(table, data, limits, categorical, order):
    """
    Applies limits to the number of categories shown for each categorical variable
    in the DataFrame, based on specified requirements.
    """
    # set the limit on the number of categorical variables
    if limits:
        levelcounts = data[categorical].nunique()

        for k, _ in levelcounts.items():
            # set the limit for the variable
            if (isinstance(limits, int)
                    and levelcounts[k] >= limits):
                limit = limits
            elif isinstance(limits, dict) and k in limits:
                limit = limits[k]
            else:
                continue

            if not order or (order and k not in order):
                # Use unique values in order of appearance, not by frequency
                unique_vals = data[k].dropna().unique()
                new_idx = [(k, '{}'.format(i)) for i in unique_vals]
            else:
                # apply order
                all_var = table.loc[k].index.unique(level='value')
                new_idx = [(k, '{}'.format(v)) for v in order[k]]
                new_idx += [(k, '{}'.format(v)) for v in all_var
                            if v not in order[k]]

            # restructure to match the original idx
            new_idx_array = np.empty((len(new_idx),), dtype=object)
            new_idx_array[:] = [tuple(i) for i in new_idx]
            orig_idx = table.index.values.copy()
            orig_idx[table.index.get_loc(k)] = new_idx_array
            table = table.reindex(orig_idx)

            # drop the rows > the limit
            table = table.drop(new_idx_array[limit:])  # type: ignore

    return table


def sort_and_reindex(table, smd, smd_table, sort, columns):
    """
    Sorts and reindexes the table to meet requirements.
    Ensures unique index before reindexing to prevent InvalidIndexError.
    """
    # sort the table rows
    sort_columns = ['Missing', 'P-Value', 'P-Value (adjusted)', 'Test']

    if smd and smd_table is not None:
        sort_columns = sort_columns + list(smd_table.columns)

    # Ensure unique index before reindexing
    if not table.index.is_unique:
        warnings.warn("Index is not unique before reindexing. Resetting index to avoid errors.")
        table = table.reset_index(drop=True)
        # After reset, create a new index based on columns order
        table.index = pd.Index([str(i) for i in range(len(table))])

    if sort and isinstance(sort, bool):
        new_index = sorted(table.index.values, key=lambda x: str(x).lower())
    elif sort and isinstance(sort, str) and (sort in sort_columns):
        try:
            new_index = table.sort_values(sort).index
        except KeyError:
            new_index = sorted(table.index.values,
                               key=lambda x: columns.index(x[0]) if isinstance(x, tuple) else 0)
            warnings.warn(f'Sort variable not found: {sort}')
    elif sort and isinstance(sort, str) and (sort not in sort_columns):
        new_index = sorted(table.index.values,
                           key=lambda x: columns.index(x[0]) if isinstance(x, tuple) else 0)
        warnings.warn(f'Sort must be in the following list: {sort}')
    else:
        # sort by the columns argument
        new_index = sorted(table.index.values,
                           key=lambda x: columns.index(x[0]) if isinstance(x, tuple) else 0)
    table = table.reindex(new_index)

    return table


def apply_order(table, order, groupby):
    """
    Applies a predefined order to rows based on specified requirements.
    May include reordering based on categorical group levels or other criteria.
    """
    # if an order is specified, apply it
    if order:
        for k in order:
            # Skip if the variable isn't present
            try:
                all_var = table.loc[k].index.unique(level='value')
            except KeyError:
                if k not in groupby:  # type: ignore
                    warnings.warn(f"Order variable not found: {k}")
                continue

            # Remove value from order if it is not present
            if [i for i in order[k] if i not in all_var]:
                rm_var = [i for i in order[k] if i not in all_var]
                order[k] = [i for i in order[k] if i in all_var]
                warnings.warn(f'Order value not found: "{k}: {rm_var}"')

            new_seq = [(k, '{}'.format(v)) for v in order[k]]
            new_seq += [(k, '{}'.format(v)) for v in all_var
                        if v not in order[k]]

            # restructure to match the original idx
            new_idx_array = np.empty((len(new_seq),), dtype=object)
            new_idx_array[:] = [tuple(i) for i in new_seq]
            orig_idx = table.index.values.copy()
            orig_idx[table.index.get_loc(k)] = new_idx_array
            table = table.reindex(orig_idx)

    return table


def mask_duplicate_values(table, optional_columns, smd, smd_table):
    """
    Masks duplicate values, ensuring that repeated values (e.g. counts of
    missing values) are only displayed once.
    """
    # only display data in first level row
    dupe_mask = table.groupby(level=[0]).cumcount().ne(0)  # type: ignore
    dupe_columns = ['Missing']

    if smd and smd_table is not None:
        optional_columns = optional_columns + list(smd_table.columns)
    for col in optional_columns:
        if col in table.columns.values:
            dupe_columns.append(col)

    table[dupe_columns] = table[dupe_columns].mask(dupe_mask).fillna('')

    return table


def create_row_labels(columns, alt_labels, label_suffix, nonnormal, 
                      min_max, categorical) -> dict:
    """
    Take the original labels for rows. Rename if alternative labels are
    provided. Append label suffix if label_suffix is True.

    Returns
    ----------
    labels : dictionary
        Dictionary, keys are original column name, values are final label.

    """
    # start with the original column names
    labels = {}
    for c in columns:
        labels[c] = str(c)  # Ensure column names are strings

    # replace column names with alternative names if provided
    if alt_labels:
        for k in alt_labels.keys():
            if k in labels:
                labels[k] = alt_labels[k]

    # append the label suffix
    if label_suffix:
        for k in labels.keys():
            if k in nonnormal:
                if min_max and k in min_max:
                    labels[k] = "{}, {}".format(labels[k],
                                                "median [min,max]")
                else:
                    labels[k] = "{}, {}".format(labels[k],
                                                "median [Q1,Q3]")
            elif k in categorical:
                labels[k] = "{}, {}".format(labels[k], "n (%)")
            else:
                if min_max and k in min_max:
                    labels[k] = "{}, {}".format(labels[k],
                                                "mean [min,max]")
                else:
                    labels[k] = "{}, {}".format(labels[k],
                                                "mean (SD)")
                
                
                

    # Ensure labels are consistent with expected test formats
    for k in labels.keys():
        labels[k] = labels[k].strip()

    return labels


def reorder_columns(table, optional_columns, groupby, order, overall):
    """
    Reorder columns for consistent, predictable formatting.
    The column order priority is:
    1. 'Missing' column first (if present)
    2. 'Overall' column (if present and overall=True)
    3. Custom order for groupby columns (if specified in order parameter)
    4. Any remaining columns
    5. Optional columns at the end (P-Value, Test, SMD, etc.)
    """
    if groupby:
        # Check if we're dealing with a MultiIndex
        if hasattr(table.columns, 'levels') and len(table.columns.levels) > 1:
            header = ['{}'.format(v) for v in table.columns.levels[1].values]
        else:
            header = ['{}'.format(v) for v in table.columns.values]
        
        # Ensure 'Missing' is always first
        cols = []
        if 'Missing' in header:
            cols = ['Missing']
            header = [x for x in header if x != 'Missing']
        
        # Ensure 'Overall' is at the beginning (after 'Missing') if overall is True
        if overall and 'Overall' in header:
            cols.append('Overall')
            header = [x for x in header if x != 'Overall']
        
        # Apply custom ordering for groupby column values if specified
        if order and (groupby in order):
            # Log what we're about to do
            ##print(f"[TABLEONE DEBUG] Applying order for '{groupby}': {order[groupby]}")
            #print(f"[TABLEONE DEBUG] Current header values: {header}")
            
            # First, convert order values to strings for comparison
            order_vals_str = ['{}'.format(val) for val in order[groupby]]
            
            # Add ordered values that exist in the header
            ordered_vals = [val for val in order_vals_str if val in header]
            
            # Add any values from header that aren't in the order list
            remaining_vals = [val for val in header if val not in ordered_vals]
            
            # Final column order
            cols.extend(ordered_vals + remaining_vals)
            #print(f"[TABLEONE DEBUG] Final column order: {cols}")
        else:
            cols.extend(header)
    else:
        cols = ['{}'.format(v) for v in table.columns.values]    # Move optional columns to the end
    for col in optional_columns:
        if col in cols:
            cols = [x for x in cols if x != col] + [col]

    # Reindex the table with the new column order
    if groupby:
        if hasattr(table.columns, 'levels') and len(table.columns.levels) > 1:
            # For MultiIndex columns
            try:
                #print(f"[TABLEONE DEBUG] Reindexing columns with MultiIndex: {cols}")
                table = table.reindex(cols, axis=1, level=1)
            except Exception as e:
                # Handle case where reindexing fails
                #print(f"[TABLEONE DEBUG] Reindexing failed: {str(e)}")
                pass
        else:
            # For regular columns
            #print(f"[TABLEONE DEBUG] Reindexing columns without MultiIndex: {cols}")
            table = table.reindex(cols, axis=1)
    else:
        table = table.reindex(cols, axis=1)

    return table


def generate_histograms(values, bins=8, clip=(1, 99)):
    """
    Generate a mini histogram using unicode blocks.

    Parameters
    ----------
    values : np.ndarray
        Numeric values.
    bins : int
        Number of bins for the histogram.
    clip : tuple of (int, int) or None, optional
        If specified, clip values to the given (lower_percentile, upper_percentile).
        For example, clip=(1, 99) clips to 1st and 99th percentiles.
        If None, no clipping is applied.

    Returns
    -------
    str
        Unicode sparkline.
    """
    if len(values) == 0:
        return ''

    if clip is not None:
        lower, upper = np.percentile(values, clip)
        values = np.clip(values, lower, upper)

    hist, _ = np.histogram(values, bins=bins)
    if hist.max() == 0:
        return ''

    blocks = '▁▂▃▄▅▆▇█'
    hist_normalized = np.floor((hist / hist.max()) * (len(blocks) - 1)).astype(int)

    return ''.join(blocks[i] for i in hist_normalized)
