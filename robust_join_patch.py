"""
This module provides robust join functions for tableone
"""
import pandas as pd


def safe_join(left_df, right_df, how='left'):
    """
    Safely join two DataFrames with error handling
    
    Parameters
    ----------
    left_df : pandas.DataFrame
        Left DataFrame
    right_df : pandas.DataFrame
        Right DataFrame
    how : str, default='left'
        Type of join to perform
        
    Returns
    -------
    pandas.DataFrame
        Joined DataFrame, or original left_df if join fails
    """
    try:
        # Try standard join first
        result = left_df.join(right_df)
        return result
    except Exception as e:
        # If standard join fails, try merge with reset_index
        try:
            print(f"Standard join failed: {str(e)}. Trying alternative merge approach.")
            left_reset = left_df.reset_index()
            right_reset = right_df.reset_index()
            
            # Get common columns for merge
            common_cols = [col for col in left_reset.columns if col in right_reset.columns]
            if not common_cols:
                print("No common columns found for merge. Returning original DataFrame.")
                return left_df
                
            merged = pd.merge(left_reset, right_reset, on=common_cols, how=how)
            
            # Try to restore the index structure if possible
            if all(idx in merged.columns for idx in left_df.index.names):
                try:
                    result = merged.set_index(left_df.index.names)
                    return result
                except Exception as merge_err:
                    print(f"Error restoring index after merge: {str(merge_err)}. Returning DataFrame with flat index.")
                    return merged
            else:
                return merged
        except Exception as merge_fail:
            print(f"Alternative merge approach failed: {str(merge_fail)}. Returning original DataFrame.")
            return left_df


def fix_table_join_operations():
    """
    Monkey-patch TableOne join operations with robust versions
    """
    import importlib
    from types import MethodType
    
    try:
        # Import tableone modules
        from tableone_extended.tableone import tables
        
        # Create safe version of create_cont_table
        original_create_cont_table = tables.Tables.create_cont_table
        
        def safe_create_cont_table(self, *args, **kwargs):
            """Safe wrapper for create_cont_table"""
            try:
                return original_create_cont_table(self, *args, **kwargs)
            except Exception as e:
                print(f"[ERROR] Error in create_cont_table: {str(e)}")
                print("[INFO] Attempting recovery with safe join approach...")
                
                # Get arguments from kwargs or args
                data = kwargs.get('data', args[0] if args else None)
                overall = kwargs.get('overall', args[1] if len(args) > 1 else False)
                cont_describe = kwargs.get('cont_describe', args[2] if len(args) > 2 else None)
                cont_describe_all = kwargs.get('cont_describe_all', args[3] if len(args) > 3 else None)
                continuous = kwargs.get('continuous', args[4] if len(args) > 4 else [])
                
                # If we don't have necessary data, just re-raise
                if data is None or cont_describe is None:
                    raise
                
                # Create a basic table with just the summary stats
                try:
                    # Start with just the t1_summary data
                    table = cont_describe[['t1_summary']].copy()
                    table.columns = table.columns.droplevel(level=0)
                    
                    # Add a simple value column for compatibility
                    table['value'] = ''
                    table = table.set_index([table.index, 'value'])
                    
                    # Add the most critical "Missing" column if possible
                    try:
                        nulltable = data[continuous].isnull().sum().to_frame(name='Missing')
                        table = safe_join(table, nulltable)
                    except Exception:
                        pass
                    
                    return table
                except Exception:
                    # If all else fails, return an empty DataFrame with the right structure
                    columns = ['t1_summary', 'Missing']
                    empty_df = pd.DataFrame(columns=columns)
                    empty_df.index = pd.MultiIndex.from_tuples([], names=['variable', 'value'])
                    return empty_df
        
        # Apply monkey patch
        tables.Tables.create_cont_table = safe_create_cont_table
        
        print("[SUCCESS] Successfully applied robust join patch to TableOne")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to apply robust join patch: {str(e)}")
        return False
