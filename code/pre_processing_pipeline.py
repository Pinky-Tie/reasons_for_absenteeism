import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pylab import rcParams


def pp_pipeline(data, remove_duplicates,change_datatypes, remove_outliers, reformat_types, standardize_missing, handle_missing):
    pass



def group_by_worker(dataframe,  demographic_cols, absence_time_col, dummies_prefix = 'Reason_'):
    ''' Created a dataframe grouped by worker ID, aggregating demographic columns by taking the first value,
    summing one-hot encoded reason columns, and calculating count, sum, and mean for absence-related metrics.
    Args:
        dataframe (pd.DataFrame): Input dataframe with worker ID as index.
    Returns:
        pd.DataFrame: Grouped dataframe by worker ID with aggregated values. '''
    reason_columns = [col for col in dataframe.columns if col.startswith(dummies_prefix)]
    absense_cols = [col for col in dataframe.columns if col not in demographic_cols and col != absence_time_col and col not in reason_columns]

    agg_dict = {}

    # aggregation rules
    for col in dataframe.columns:
        #if col is demographic, use mode
        if col in demographic_cols:
            agg_dict[col] = lambda x: x.mode()[0]
        #if col is a metric, use sum
        elif col in absense_cols or col in reason_columns:
            agg_dict[col] = 'sum'
        #if col is the main abscence time column, use count, sum, mean
        elif col == absence_time_col:
            agg_dict[col] = ['count', 'sum', 'mean']

    
    # Group by worker ID and aggregate
    grouped = dataframe.groupby(level=0).agg(agg_dict)
    
    # Flatten column names if they became multi-index
    if isinstance(grouped.columns, pd.MultiIndex):
        grouped.columns = [f"{col[0]}_{col[1]}" if col[1] != 'first' else col[0] 
                         for col in grouped.columns]
    
    return grouped
    

