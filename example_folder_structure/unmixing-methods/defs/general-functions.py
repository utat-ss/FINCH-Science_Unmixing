import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib as mpl
from sklearn.model_selection import train_test_split
import shap
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


MIN_WAVELENGTH = 900
MAX_WAVELENGTH = 1700
RANDOM_STATE = 42
NPV_COLUMN = 1


def prepare_ds(file: str = 'simpler_data.csv', test_size: float = 0.2) -> np.ndarray:
    """
    Return the normalized dataset in  <file> as an array, split with test size <test size>.

    Note: Normalization to mean 0 and standard deviation 1 is done based on the mean and st. dev.
    of the training data only to avoid data leakage, but is applied to both training and testing.

    The returns are all pandas dataframes with the following shapes:
    x_train: (1378, 81)
    x_test: (345, 81)
    y_train: (1378, 3)
    y_test: (345, 3)
    """

    ds = pd.read_csv(file)
    wavelength_cols = [col for col in ds.columns if
                       col.isdigit() and MIN_WAVELENGTH <= int(col) <= MAX_WAVELENGTH]
    target_cols = ['gv_fraction', 'npv_fraction', 'soil_fraction']
    x = ds[wavelength_cols]
    y = ds[target_cols]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=RANDOM_STATE)
    mean, sd = x_train.mean(axis=0), x_train.std(axis=0)
    x_train, x_test = (x_train - mean) / sd, (x_test - mean) / sd
    return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


def evaluate_model(predicted_value: np.ndarray, true_value: np.ndarray, p: bool) -> tuple[float]:
    """
    Return (and print if <p>) the rmse, r2 and r2 of y = x of <true_value> vs <predicted_value>
    """
    rmse = np.sqrt(np.mean((true_value - predicted_value) ** 2))
    rmse_npv = np.sqrt(np.mean((true_value[:, NPV_COLUMN] - predicted_value[:, NPV_COLUMN]) ** 2))
    r2_npv = r2_score(true_value[:, NPV_COLUMN], predicted_value[:, NPV_COLUMN])
    r2 = r2_score(true_value, predicted_value)
    ss_res = np.sum((predicted_value - true_value) ** 2)
    ss_tot = np.sum((true_value - np.mean(true_value)) ** 2)
    r2_y_eq_x = 1 - ss_res / ss_tot
    ss_res = np.sum((predicted_value[:, NPV_COLUMN] - true_value[:, NPV_COLUMN]) ** 2)
    ss_tot = np.sum((true_value[:, NPV_COLUMN] - true_value[:, NPV_COLUMN]) ** 2)
    r2_y_eq_x_npv = 1 - ss_res / ss_tot
    if p:
        print(f"RMSE: {rmse:.4f}")
        print(f"RMSE (NPV): {rmse_npv:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"R² (NPV): {r2_npv:.4f}")
        print(f"R² (y = x): {r2_y_eq_x:.4f}")
        print(f"R² (y = x) (NPV): {r2_y_eq_x_npv:.4f}")
    return (rmse, rmse_npv, r2_npv, r2, r2_y_eq_x, r2_y_eq_x_npv)


def plot_preds(ab_true: np.ndarray, ab_pred: np.ndarray, npv_bestfit: bool = True,
               save_path: str = None) -> None:
    """
    Function to plot predicted vs true abundances for three classes. Optionally bestfits for the npv.

    Args:
        ab_true (np.ndarray): True abundances, shape (n_samples, 3)
        ab_pred (np.ndarray): Predicted abundances, shape (n_samples, 3)
        npv_bestfit (bool): If True, use bestfit for NPV predictions
        save_path (str): If provided, save the plot to this path as a SVG
    """
    assert (type(ab_true) is np.ndarray) and (
                type(ab_pred) is np.ndarray), "ab_true and ab_pred must be numpy arrays"
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
            marker=marker
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
            label='NPV Bestfit'
        )

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("True Abundance")
    plt.ylabel("Predicted Abundance")
    plt.title("Predicted vs True Abundances")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='svg')
    plt.show()
