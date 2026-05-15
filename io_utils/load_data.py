import pandas as pd
import numpy as np


def load_df(
    file_path: str,
    sheet_name: str = 0,
    skiprows: None | int = None,
    usecols: None | list = None,
) -> pd.DataFrame | None:
    try:
        data: pd.DataFrame = pd.read_excel(
            file_path, sheet_name, skiprows=skiprows, usecols=usecols
        )
        print(f"'{file_path}' loaded successfully...")
        for col in data.select_dtypes(include="object").columns:
            data[col] = data[col].str.strip()
        data.columns = data.columns.str.strip()
        data: pd.DataFrame = data.replace(r"^\s*$", np.nan, regex=True)
        return data
    except FileNotFoundError:
        print(
            f"Error: The specified file '{file_path}' was not found. Please check the file path."
        )
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading '{file_path}': {e}")
        return None
