# util.py
import pandas as pd
import numpy as np

def replace_invalid_values(x):
    """Replace invalid entries with NaN."""
    if isinstance(x, (str, int, float)):
        if str(x).strip().lower() in ['na', 'nan', 'not applicable', 'n/a', 'n.a.', 'null', 'empty', 'blank']:
            return np.nan
    return x

def clean_dataframe(df):
    df_update = df.copy()
    summary = {
        'numeric_columns_filled': {},
        'numeric_outliers_capped': {},
        'categorical_columns_filled': {},
        'categorical_columns_removed': [],
        'datetime_columns_filled': {},
        'rows_removed': 0,
        'columns_removed': 0
    }

    # Remove empty rows and columns
    rows_before = df_update.shape[0]
    df_update.dropna(how='all', inplace=True)
    summary['rows_removed'] = rows_before - df_update.shape[0]

    columns_before = df_update.shape[1]
    df_update.dropna(axis=1, how='all', inplace=True)
    summary['columns_removed'] = columns_before - df_update.shape[1]

    # Clean numeric columns
    for col in df_update.select_dtypes(include=[np.number]).columns:
        df_update[col] = df_update[col].map(replace_invalid_values)
        missing_count = df_update[col].isnull().sum()
        if missing_count > 0:
            mean_value = df_update[col].mean()
            df_update[col].fillna(mean_value, inplace=True)
            summary['numeric_columns_filled'][col] = missing_count

        # Detect and cap outliers
        lower_bound = df_update[col].quantile(0.01)
        upper_bound = df_update[col].quantile(0.99)
        outliers_lower = (df_update[col] < lower_bound).sum()
        outliers_upper = (df_update[col] > upper_bound).sum()
        if outliers_lower > 0 or outliers_upper > 0:
            df_update[col] = np.clip(df_update[col], lower_bound, upper_bound)
            summary['numeric_outliers_capped'][col] = {
                'lower_capped': outliers_lower,
                'upper_capped': outliers_upper
            }

    # Clean categorical columns
    for col in df_update.select_dtypes(include=['object']).columns:
        df_update[col] = df_update[col].map(replace_invalid_values).astype(str).str.strip()
        missing_percentage = df_update[col].isnull().mean()
        if missing_percentage > 0.9:
            df_update.drop(columns=[col], inplace=True)
            summary['categorical_columns_removed'].append(col)
        else:
            missing_count = df_update[col].isnull().sum()
            if missing_count > 0:
                df_update[col].fillna('Not Specified', inplace=True)
                summary['categorical_columns_filled'][col] = missing_count

    # Clean datetime columns
    for col in df_update.select_dtypes(include=['datetime']).columns:
        df_update[col] = pd.to_datetime(df_update[col], errors='coerce')
        missing_count = df_update[col].isnull().sum()
        if missing_count > 0:
            mode_value = df_update[col].mode()[0]
            df_update[col].fillna(mode_value, inplace=True)
            summary['datetime_columns_filled'][col] = missing_count

    # Build the markdown summary string
    summary_md = "**Data Cleaning Result:**\n\n"

    if summary['numeric_columns_filled']:
        filled_cols = ', '.join(
            [f"{col} ({count} values)" for col, count in summary['numeric_columns_filled'].items()]
        )
        summary_md += f"- Numeric columns with missing values filled using the column mean:\n  {filled_cols}\n\n"

    if summary['numeric_outliers_capped']:
        capped_cols = ', '.join(
            [f"{col} (lower: {caps['lower_capped']}, upper: {caps['upper_capped']})"
             for col, caps in summary['numeric_outliers_capped'].items()]
        )
        summary_md += f"- Numeric columns with outliers capped between the 1st and 99th percentiles:\n  {capped_cols}\n\n"

    if summary['categorical_columns_filled']:
        filled_cats = ', '.join(
            [f"{col} ({count} values)" for col, count in summary['categorical_columns_filled'].items()]
        )
        summary_md += f"- Categorical columns with missing values filled with 'Not Specified':\n  {filled_cats}\n\n"

    if summary['categorical_columns_removed']:
        removed_cols = ', '.join(summary['categorical_columns_removed'])
        summary_md += f"- Categorical columns removed due to over 90% missing data:\n  {removed_cols}\n\n"

    if summary['datetime_columns_filled']:
        filled_dates = ', '.join(
            [f"{col} ({count} values)" for col, count in summary['datetime_columns_filled'].items()]
        )
        summary_md += f"- Datetime columns with missing values filled using the column mode:\n  {filled_dates}\n\n"

    summary_md += f"- Total number of rows removed: {summary['rows_removed']}\n"
    summary_md += f"- Total number of columns removed: {summary['columns_removed']}\n\n"
    summary_md += "Next, we review and standardize categorical fields, identifying any unreasonable values.\n"

    return df_update, summary_md
