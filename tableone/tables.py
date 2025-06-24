from typing import Optional

import numpy as np
import pandas as pd

from .statistics import Statistics
from .exceptions import InputError, non_continuous_warning


class Tables:
    """
    Create and store intermediate tables used to create Table 1.

    Usage:

    self.tables = Tables()
    self.tables._create_htest_table()
    self.tables.htest_table
    """
    def __init__(self):
        """
        Initialize the Tables class.
        """
        self.statistics = Statistics()

    def create_htest_table(self, data: pd.DataFrame,
                           continuous,
                           categorical,
                           nonnormal,
                           groupby,
                           groupbylvls,
                           htest,
                           pval,
                           pval_adjust,
                           ttest_equal_var) -> pd.DataFrame:
        """
        Create a table containing P-Values for significance tests. Add features
        of the distributions and the P-Values to the dataframe.

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.

        Returns
        ----------
            df : pandas DataFrame
                A table containing the P-Values, test name, etc.
        """
        # list features of the variable e.g. matched, paired, n_expected
        df = pd.DataFrame(index=continuous+categorical,
                          columns=['continuous', 'nonnormal',
                                   'min_observed', 'P-Value', 'Test'])

        df.index = df.index.rename('variable')

        df['continuous'] = np.where(df.index.isin(continuous), True, False)
        df['nonnormal'] = np.where(df.index.isin(nonnormal), True, False)

        # list values for each variable, grouped by groupby levels
        min_observed = 0
        catlevels = None

        for v in df.index:
            is_continuous = df.loc[v]['continuous']
            is_categorical = ~df.loc[v]['continuous']
            is_normal = ~df.loc[v]['nonnormal']

            # if continuous, group data into list of lists
            if is_continuous:
                catlevels = None
                grouped_data = {}
                for s in groupbylvls:
                    lvl_data = data.loc[data[groupby] == s, v]
                    # coerce to numeric and drop non-numeric data
                    lvl_data = lvl_data.apply(pd.to_numeric,
                                              errors='coerce').dropna()
                    # append to overall group data
                    grouped_data[s] = lvl_data.values
                min_observed = min([len(x) for x in grouped_data.values()])
            # if categorical, create contingency table
            elif is_categorical:
                catlevels = sorted(data[v].astype('category').cat.categories)
                cross_tab = pd.crosstab(data[groupby].rename('_groupby_var_'), data[v])
                min_observed = cross_tab.sum(axis=1).min()
                grouped_data = cross_tab.T.to_dict('list')

            # minimum number of observations across all levels
            df.loc[v, 'min_observed'] = min_observed  # type: ignore

            # compute pvalues
            warning_msg = None
            (df.loc[v, 'P-Value'],
             df.loc[v, 'Test'],
             warning_msg) = self.statistics._p_test(v, grouped_data, is_continuous, is_categorical,  # type: ignore
                                                    is_normal,  min_observed, htest, ttest_equal_var)  # type: ignore

            # TODO: Improve method for handling these warnings.
            # Write to logfile?
            #
            # if warning_msg:
            #     try:
            #         self._warnings[warning_msg].append(v)
            #     except KeyError:
            #         self._warnings[warning_msg] = [v]

        # correct for multiple testing
        if pval and pval_adjust:
            adjusted = self.statistics.multipletests(df['P-Value'],
                                                     alpha=0.05,
                                                     method=pval_adjust)
            df['P-Value (adjusted)'] = adjusted[1]
            df['adjust method'] = pval_adjust

        return df

    def create_smd_table(self,
                         groupbylvls,
                         continuous,
                         categorical,
                         cont_describe,
                         cat_describe) -> pd.DataFrame:
        """
        Create a table containing pairwise Standardized Mean Differences
        (SMDs).

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.

        Returns
        ----------
            df : pandas DataFrame
                A table containing pairwise standardized mean differences
                (SMDs).
        """
        # create the SMD table
        permutations = [sorted((x, y),
                        key=lambda f: groupbylvls.index(f))
                        for x in groupbylvls
                        for y in groupbylvls if x is not y]

        p_set = set(tuple(x) for x in permutations)

        colname = 'SMD ({0},{1})'
        columns = [colname.format(x[0], x[1]) for x in p_set]
        df = pd.DataFrame(index=continuous+categorical, columns=columns)
        df.index = df.index.rename('variable')

        for p in p_set:
            try:
                for v in cont_describe.index:
                    smd, _ = self.statistics._cont_smd(
                                mean1=cont_describe['mean'][p[0]].loc[v],
                                mean2=cont_describe['mean'][p[1]].loc[v],
                                sd1=cont_describe['std'][p[0]].loc[v],
                                sd2=cont_describe['std'][p[1]].loc[v],
                                n1=cont_describe['count'][p[0]].loc[v],
                                n2=cont_describe['count'][p[1]].loc[v],
                                unbiased=False)
                    df.loc[v, colname.format(p[0], p[1])] = smd
            except AttributeError:
                pass

            try:
                for v, _ in cat_describe.groupby(level=0):
                    smd, _ = self.statistics._cat_smd(
                        prop1=cat_describe.loc[[v]]['percent'][p[0]].values/100,
                        prop2=cat_describe.loc[[v]]['percent'][p[1]].values/100,
                        n1=cat_describe.loc[[v]]['freq'][p[0]].sum(),
                        n2=cat_describe.loc[[v]]['freq'][p[1]].sum(),
                        unbiased=False)
                    df.loc[v, colname.format(p[0], p[1])] = smd  # type: ignore
            except AttributeError:
                pass

        return df

    def format_cat(self, row, col, decimals) -> str:
        """
        Format values to n decimal places.
        """
        var = row.name[0]
        if var in decimals:
            n = decimals[var]  # type: ignore
        else:
            n = 1
        f = '{{:.{}f}}'.format(n)

        return f.format(row[col])

    def create_cat_describe(self,
                            data: pd.DataFrame,
                            categorical,
                            decimals,
                            row_percent,
                            include_null,
                            groupby: Optional[str] = None,
                            groupbylvls: Optional[list] = None,
                            include_nulls_in_percent: bool = True
                            ) -> pd.DataFrame:
        """
        Describe the categorical data.

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.
            groupby : Str
                Variable to group by.
            groupbylvls : List
                List of levels in the groupby variable.

        Returns
        ----------
            df_cat : pandas DataFrame
                Summarise the categorical variables.
        """
        group_dict = {}

        cat_slice = data[categorical].copy()

        # Use category order from dtype if available, else fallback to sorted unique values
        def get_levels(col):
            s = cat_slice[col]
            if pd.api.types.is_categorical_dtype(s):
                return list(s.cat.categories)
            else:
                return sorted(set(s.dropna().astype(str)))

        all_levels = {col: get_levels(col) for col in categorical}

        # Build full multi-index with all combinations
        full_index = pd.MultiIndex.from_tuples(
            [(col, val) for col in categorical for val in all_levels[col]],
            names=['variable', 'value']
        )

        for g in groupbylvls:  # type: ignore
            if groupby:
                df = cat_slice.loc[data[groupby] == g, categorical]
            else:
                df = cat_slice.copy()

            # create n column
            # must be done before converting values to strings
            ct = df.count().to_frame(name='n')
            ct.index.name = 'variable'

            if include_null:
                # create an empty Missing column for display purposes
                nulls = pd.DataFrame('', index=df.columns, columns=['Missing'])
                nulls.index.name = 'variable'
            else:
                # Count and display null count
                nulls = df.isnull().sum().to_frame(name='Missing')
                nulls.index.name = 'variable'

            # Convert to str to handle int converted to boolean in the index.
            # Also avoid nans.
            for column in df.columns:
                df[column] = [str(row) if not pd.isnull(row)
                              else None for row in df[column].values]
                cat_slice[column] = [str(row) if not pd.isnull(row)
                                     else None for row
                                     in cat_slice[column].values]

            # create a dataframe with freq, proportion
            value_name = self._get_unique_value_name(df)
            df = (
                df.melt(value_name=value_name)
                .groupby(['variable', value_name])
                .size()
                .reindex(full_index, fill_value=0)
                .to_frame(name='freq')
            )

            # Calculate denominator for percent
            if include_nulls_in_percent:
                denom = df.groupby(level=0).freq.sum()
            else:
                # Exclude nulls from denominator
                denom = df.groupby(level=0).apply(lambda x: x.loc[[i for i in x.index if i[1] != 'None' and i[1] != 'Not Reported'], 'freq'].sum())
            df['percent'] = df['freq'].div(denom, level=0).astype(float) * 100

            # add row percent
            value_name2 = self._get_unique_value_name(cat_slice[categorical], base_name='melt_value')
            full_counts = (
                cat_slice[categorical]
                .melt(value_name=value_name2)
                .groupby(['variable', value_name2])
                .size()
            )

            df['percent_row'] = df.index.map(
                lambda idx: df.at[idx, 'freq'] / full_counts.get(idx, np.nan) * 100
            )

            # set number of decimal places for percent
            if isinstance(decimals, int):
                n = decimals
                f = '{{:.{}f}}'.format(n)
                df['percent_str'] = df['percent'].astype(float).map(f.format)
                df['percent_row_str'] = df['percent_row'].astype(float).map(
                    f.format)
            elif isinstance(decimals, dict):
                df.loc[:, 'percent_str'] = df.apply(self.format_cat, axis=1,
                                                    args=['percent', decimals])
                df.loc[:, 'percent_row_str'] = df.apply(self.format_cat, axis=1,
                                                        args=['percent_row', decimals])
            else:
                n = 1
                f = '{{:.{}f}}'.format(n)
                df['percent_str'] = df['percent'].astype(float).map(f.format)
                df['percent_row_str'] = df['percent_row'].astype(float).map(
                    f.format)            # join count column with error handling
            try:
                df = df.join(ct)
            except Exception as e:
                print(f"[WARNING] Error joining count column: {str(e)}. Skipping join.")

            # only save null count to the first category for each variable
            # do this by extracting the first category from the df row index
            levels = df.reset_index()[['variable',
                                       'value']].groupby('variable').first()
            # add this category to the nulls table
            nulls = nulls.join(levels)
            nulls = nulls.set_index('value', append=True)            # join nulls to categorical with error handling
            try:
                df = df.join(nulls)
            except Exception as e:
                print(f"[WARNING] Error joining nulls table: {str(e)}. Skipping join.")

            # add summary column
            if row_percent:
                df['t1_summary'] = (df.freq.map(str) + ' ('
                                    + df.percent_row_str.map(str)+')')
            else:
                df['t1_summary'] = (df.freq.map(str) + ' ('
                                    + df.percent_str.map(str)+')')

            # Remove percentage for rows with zero count
            zero_mask = df['freq'] == 0
            df.loc[zero_mask, 't1_summary'] = df.loc[zero_mask, 'freq'].astype(str)

            # If not including nulls in percent, remove percent for null rows in summary
            if not include_nulls_in_percent:
                null_labels = ['None', 'Not Reported']  # Add more if needed or pass as parameter
                for null_label in null_labels:
                    mask = df.index.get_level_values(1) == null_label
                    if row_percent:
                        df.loc[mask, 't1_summary'] = df.loc[mask, 'freq'].astype(str)
                    else:
                        df.loc[mask, 't1_summary'] = df.loc[mask, 'freq'].astype(str)

            # add to dictionary
            group_dict[g] = df

        df_cat = pd.concat(group_dict, axis=1)
        # ensure the groups are the 2nd level of the column index
        if df_cat.columns.nlevels > 1:
            df_cat = df_cat.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)

        return df_cat

    def create_cont_describe(self,
                             data: pd.DataFrame,
                             ddof,
                             t1_summary,
                             dip_test,
                             tukey_test,
                             normal_test,
                             continuous,
                             groupby: Optional[str] = None) -> pd.DataFrame:
        """
        Describe the continuous data.

        Parameters
        ----------
            data : pandas DataFrame
                The input dataset.

        Returns
        ----------
            df_cont : pandas DataFrame
                Summarise the continuous variables.
        """
        # wrapper for std with ddof
        def std(x):
            return self.statistics._std(x, ddof)

        aggfuncs = ['count', 'mean', 'median', std,
                    self.statistics._q25, self.statistics._q75,
                    'min', 'max', t1_summary]

        if dip_test:
            aggfuncs.append(self.statistics._hartigan_dip)

        if tukey_test:
            aggfuncs.append(self.statistics._outliers)
            aggfuncs.append(self.statistics._far_outliers)

        if normal_test:
            aggfuncs.append(self.statistics._normality)

        # coerce continuous data to numeric
        cont_data = data[continuous].apply(pd.to_numeric, errors='coerce')
        # check all data in each continuous column is numeric
        bad_cols = cont_data.count() != data[continuous].count()
        bad_cols = cont_data.columns[bad_cols]
        if len(bad_cols) > 0:
            msg = ("The following continuous column(s) have "
                   "non-numeric values: {variables}. Either specify the "
                   "column(s) as categorical or remove the "
                   "non-numeric values.").format(variables=bad_cols.values)
            raise InputError(msg)        # check for coerced column containing all NaN to warn user
        for column in cont_data.columns[cont_data.count() == 0]:
            non_continuous_warning(column)
            
        if groupby:
            # add the groupby column back, ensuring both dataframes have compatible indexes
            try:
                # Make sure groupby column exists in data
                if groupby in data.columns:
                    # Ensure indexes are aligned before merge
                    cont_data_index = cont_data.index
                    data_subset = data.loc[data.index.isin(cont_data_index), [groupby]]
                    cont_data = cont_data.merge(data_subset, left_index=True, right_index=True, how='left')
                    
                    # group and aggregate data
                    df_cont = pd.pivot_table(cont_data, columns=[groupby], aggfunc=aggfuncs)
                else:
                    # Handle case where groupby column doesn't exist
                    print(f"[WARNING] Groupby column '{groupby}' not found in data. Using simple aggregation.")
                    df_cont = cont_data.apply(aggfuncs).T
                    df_cont.columns.name = 'Overall'
                    df_cont.columns = pd.MultiIndex.from_product([df_cont.columns, ['Overall']])
            except Exception as e:
                print(f"[ERROR] Error in pivot table creation: {str(e)}. Using simple aggregation.")
                # Fall back to simple aggregation without groupby
                df_cont = cont_data.apply(aggfuncs).T
                df_cont.columns.name = 'Overall'
                df_cont.columns = pd.MultiIndex.from_product([df_cont.columns, ['Overall']])
        else:
            # if no groupby, just add single group column
            df_cont = cont_data.apply(aggfuncs).T  # type: ignore
            df_cont.columns.name = 'Overall'
            df_cont.columns = pd.MultiIndex.from_product([df_cont.columns, ['Overall']])

        df_cont.index = df_cont.index.rename('variable')

        # remove prefix underscore from column names (e.g. _std -> std)
        agg_rename = df_cont.columns.levels[0]  # type: ignore
        agg_rename = [x[1:] if x[0] == '_' else x for x in agg_rename]
        df_cont.columns = df_cont.columns.set_levels(agg_rename, level=0)  # type: ignore

        return df_cont

    def create_cont_table(self,
                          data,
                          overall,
                          cont_describe,
                          cont_describe_all,
                          continuous,
                          pval,
                          pval_adjust,
                          htest_table,
                          smd,
                          smd_table,
                          groupby
                          ) -> pd.DataFrame:
        """
        Create tableone for continuous data.

        Returns
        ----------
        table : pandas DataFrame
            A table summarising the continuous variables.
        """
        # remove the t1_summary level
        table = cont_describe[['t1_summary']].copy()
        table.columns = table.columns.droplevel(level=0)        # add a column of null counts as 1-count() from previous function
        nulltable = data[continuous].isnull().sum().to_frame(name='Missing')
        try:
            table = table.join(nulltable)
        # if columns form a CategoricalIndex, need to convert to string first
        except TypeError:
            table.columns = table.columns.astype(str)
            try:
                table = table.join(nulltable)
            except Exception as e:
                print(f"[WARNING] Error joining nulltable: {str(e)}. Skipping nulltable join.")
        except Exception as e:
            print(f"[WARNING] Error joining nulltable: {str(e)}. Skipping nulltable join.")

        # add an empty value column, for joining with cat table
        table['value'] = ''
        table = table.set_index([table.index, 'value'])  # type: ignore

        # add pval column
        if pval and pval_adjust:
            table = table.join(htest_table[['P-Value (adjusted)', 'Test']])
        elif pval:
            table = table.join(htest_table[['P-Value', 'Test']])

        # add standardized mean difference (SMD) column/s
        if smd:
            table = table.join(smd_table)        # join the overall column if needed
        if groupby and overall:
            try:
                if 'Overall' in cont_describe_all['t1_summary']:  # Check if 'Overall' exists
                    overall_data = pd.concat([cont_describe_all['t1_summary'].Overall], 
                                            axis=1, keys=["Overall"])
                    
                    # Ensure indexes are compatible before joining
                    if table.index.equals(overall_data.index):
                        table = table.join(overall_data)
                    else:
                        # Use a safer merging approach when indexes don't match
                        print(f"[INFO] Index mismatch detected. Using merge instead of join for overall data.")
                        table_reset = table.reset_index()
                        overall_reset = overall_data.reset_index()
                        # Determine merge keys based on columns present
                        merge_keys = ['variable', 'value'] if 'value' in table_reset.columns and 'value' in overall_reset.columns else ['variable']
                        merged = pd.merge(table_reset, overall_reset, on=merge_keys, how='left')
                        table = merged.set_index(merge_keys)

                        # Ensure index is a MultiIndex with exactly ['variable', 'value'] as names
                        if not isinstance(table.index, pd.MultiIndex) or table.index.nlevels != 2:
                            # If only 'variable' is present, add a 'value' column of empty strings
                            if 'value' not in table.columns:
                                table['value'] = ''
                            table = table.reset_index(drop=False)
                            table = table.set_index(['variable', 'value'])
                        # Always set index names to the expected
                        table.index.set_names(['variable', 'value'], inplace=True)
            except Exception as e:
                print(f"[WARNING] Error joining overall column for continuous data: {str(e)}. Skipping join.")

        # After building the continuous table, add a missingness row per variable when grouping
        if groupby:
            # Compute missing counts per group level
            grp_miss = data.groupby(groupby)[continuous].apply(lambda df: df.isnull().sum()).T
            # Build missingness rows
            missing_rows = []
            missing_idx = []
            for var in continuous:
                # overall missing count
                overall_miss = data[var].isnull().sum()
                # All values as strings
                row = {lvl: str(int(grp_miss.loc[var, lvl])) for lvl in grp_miss.columns}
                if overall:
                    row['Overall'] = str(int(overall_miss))
                # blank for other columns
                for col in table.columns:
                    if col not in row:
                        row[col] = ''
                missing_rows.append(row)
                missing_idx.append((var, 'Not Reported'))  # Use 'Not Reported' as the value for missingness
            # Create missingness DataFrame (all strings)
            miss_df = pd.DataFrame(missing_rows,
                                   index=pd.MultiIndex.from_tuples(missing_idx, names=table.index.names),
                                   dtype=object)
            # Align columns
            miss_df = miss_df.reindex(columns=table.columns)
            # Concatenate missing rows
            table = pd.concat([table, miss_df])
            # Sort lexicographically so '' comes before 'Missing' under each variable
            table = table.sort_index()

        return table

    def create_cat_table(self,
                         data,
                         overall,
                         cat_describe,
                         categorical,
                         include_null,
                         pval,
                         pval_adjust,
                         htest_table,
                         smd,
                         smd_table,
                         groupby,
                         cat_describe_all):
        """
        Create table one for categorical data.

        Returns
        ----------
        table : pandas DataFrame
            A table summarising the categorical variables.
        """
        table = cat_describe['t1_summary'].copy()

        if include_null:
            isnull = pd.DataFrame(index=categorical, columns=['Missing'])
            isnull['Missing'] = ''
            isnull.index.rename('variable', inplace=True)
        else:
            # add the total count of null values across all levels
            isnull = data[categorical].isnull().sum().to_frame(name='Missing')
            isnull.index = isnull.index.rename('variable')

        try:
            table = table.join(isnull)
        # if columns form a CategoricalIndex, need to convert to string first
        except TypeError:
            table.columns = table.columns.astype(str)
            table = table.join(isnull)

        # add pval column
        if pval and pval_adjust:
            table = table.join(htest_table[['P-Value (adjusted)', 'Test']])
        elif pval:
            table = table.join(htest_table[['P-Value', 'Test']])

        # add standardized mean difference (SMD) column/s
        if smd:
            table = table.join(smd_table)        # join the overall column if needed
        if groupby and overall:
            try:
                overall_data = pd.concat([cat_describe_all['t1_summary'].Overall],
                                         axis=1, keys=["Overall"])
                table = table.join(overall_data)
            except Exception as e:
                print(f"[WARNING] Error joining overall column for categorical data: {str(e)}. Skipping join.")

        return table

    def _get_unique_value_name(self, df, base_name='melt_value'):
        value_name = base_name
        i = 1
        while value_name in df.columns:
            value_name = f"{base_name}_{i}"
            i += 1
        return value_name
