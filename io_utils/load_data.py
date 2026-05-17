import pandas as pd
import numpy as np


def load_df(
    file_path: str,
    sheet_name: str = 0,
    skiprows: None | int = None,
    usecols: None | list = None,
) -> pd.DataFrame | None:
    try:
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            skiprows=skiprows,
            usecols=usecols,
            dtype=str  # ensures consistent string cleaning
        )

        print(f"'{file_path}' loaded successfully...")

        # Strip whitespace from all string cells + column names in one pass
        df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
        df.columns = df.columns.str.strip()

        # Replace empty strings or whitespace-only with NaN
        df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

        return df

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

    except Exception as e:
        print(f"Unexpected error while loading '{file_path}': {e}")
        return None