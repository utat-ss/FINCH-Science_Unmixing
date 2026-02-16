import pandas as pd
import numpy as np
import pathlib
import re
import scipy
from sklearn import linear_model
import matplotlib as mpl
plt = mpl.pyplot

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

# Copied from DL's plotting.py file
def plot_preds(ab_true:(np.ndarray), ab_pred:(np.ndarray), save_path:(str), model_name:(str), npv_bestfit:(bool)=True) -> None:
    """
    Function to plot predicted vs true abundances for three classes. Optionally bestfits for the npv.

    Args:
        ab_true (np.ndarray): True abundances, shape (n_samples, 3)
        ab_pred (np.ndarray): Predicted abundances, shape (n_samples, 3)
        save_path (str): Save the plot to this path as a SVG
        model_name (str): The name of the model that was used
        npv_bestfit (bool): If True, use bestfit for NPV predictions
    """
    assert (type(ab_true) is np.ndarray) and (type(ab_pred) is np.ndarray), "ab_true and ab_pred must be numpy arrays"
    assert ab_true.shape == ab_pred.shape, "ab_true and ab_pred must have the same shape"

    # Formatting the plot
    mpl.rcParams['font.family'] = 'Times New Roman'
    colors = ['green', 'orange', 'saddlebrown']
    labels = ['GV', 'NPV', 'Soil']
    markers = ['o', '^', 's']
    
    # Start plot
    plt.figure(figsize=(8, 6))
    
    # The regular scatter plots
    for i, (label, color, marker) in enumerate(zip(labels, colors, markers)):
        plt.scatter(
            ab_true[:, i],
            ab_pred[:, i],
            label=label,
            alpha=0.6,
            edgecolor='k',
            color=color,
            marker=marker,
            zorder=2
            )

    # Bestfit for npv
    if npv_bestfit:
        m, c = np.polyfit(ab_true[:, 1], ab_pred[:, 1], 1)
        x_fit = np.linspace(0, 1, 100)
        y_fit = m * x_fit + c
        plt.plot(
            x_fit,
            y_fit,
            color='darkgoldenrod',
            linestyle='--',
            label='NPV Bestfit',
            zorder=3
            )

    plt.plot([0, 1], [0, 1], 'k--', zorder=1)
    plt.xlabel("True Abundance")
    plt.ylabel("Predicted Abundance")
    plt.title(f"{model_name}, Predicted vs True Abundances")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.show()