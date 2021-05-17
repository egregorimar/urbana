"""Tools to handle with datasets."""

import pandas as pd

from urbana.constants import DIR_DATA_RAW


def merge_datasets(master_dataset, new_dataset):
    """Merge two datasets

    Args:
        master_dataset (pd.DataFrame): the master dataset where the new dataset
             will be appended
        new_dataset (pd.DataFrame): the new dataset to merge to the master_dataset

    Returns:
        pd.DataFrame: the master dataset with the new dataset appended
    """
    return pd.merge(
        left=master_dataset,
        right=new_dataset,
        how="left",
        left_index=True,
        right_index=True,
    )


def get_insideairbnb_data_from_url(year: int, month: int, day: int) -> pd.DataFrame:
    """Get data from insideairbnb.

    Ref: http://insideairbnb.com/get-the-data.html

    Args:
        year (int): Year (4-digits formatted)
        month (int): Month
        day (int): Day

    Returns:
        pd.DataFrame: dataframe loaded from insideairbnb
    """
    airbnb_url = (
        f"http://data.insideairbnb.com/spain/catalonia/barcelona/{year}-{month:02}-{day:02}"
        "/data/listings.csv.gz"
    )
    return pd.read_csv(airbnb_url)


def get_insideairbnb_data(year: int, month: int, day: int) -> pd.DataFrame:
    """Get data from insideairbnb.

    The data is loaded either from the data folder or from the insideairbnb url.

    Ref: http://insideairbnb.com/get-the-data.html

    Args:
        year (int): Year (4-digits formatted)
        month (int): Month
        day (int): Day

    Returns:
        pd.DataFrame: dataframe loaded from insideairbnb
    """
    insideairbnb_filename = f"{year}-{month:02}-{day:02}.csv"
    try:
        df = pd.read_csv(DIR_DATA_RAW / "insideairbnb" / insideairbnb_filename)
    except FileNotFoundError:
        df = get_insideairbnb_data_from_url(year, month, day)
    return df
