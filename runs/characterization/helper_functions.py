import pandas as pd
import numpy as np
import pathlib
import re
import scipy
from sklearn import linear_model
import matplotlib.pyplot as plt

"""
Helper functions used by dataset_characterization.ipynb and other notebooks.

Author: Andrei Akopian
"""

def open_file(filename: str):
    """
    Return the dataset under the filename.

    Has specific handlers for specific spreadsheet file formats.
    """
    path = pathlib.PurePath(filename)
    file_format = path.suffix
    parsing_functions = {
        ".csv" : pd.read_csv,
    }
    return parsing_functions[file_format](filename)

def take_subset(df: pd.DataFrame, start: int, end: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return 2 pandas dataframes, from the input dataframe 
    (input expectecd to be a spreadsheet with columns and structure similar to original_data.csv)
    Inputs:
    - start: integer of the first wavelength in the subset
    - end: integer of the last wavelength in the subset

    - one dataframe with abundance columns only
    - one dataframe with reflectances and wavelength columns from start to end

    return (npv_fractions, spectra, spectra_sources)
    """

    columns = df.columns.to_list()
    wanted = []
    for c in columns:
        if c.isdigit():
            if start<=int(c)<=end:
                wanted.append(c)
    abundances = df[["npv_fraction","gv_fraction","soil_fraction"]]
    spectra = df[wanted]
    return abundances, spectra

def simple_histogram(data, title="Title", x_label="x-axis" ,y_label='y-axis', bins=10):
    """
    Create matplotlib histogram of the provided data.
    Made for informal visualizations, and common arguments can be changed easily.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.hist(data,bins=bins)