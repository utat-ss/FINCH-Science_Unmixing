import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

def plot_preds(ab_true:(np.ndarray), ab_pred:(np.ndarray), npv_bestfit:(bool)=True, save_path:(str)=None) -> None:
    """
    Function to plot predicted vs true abundances for three classes. Optionally bestfits for the npv.

    Args:
        ab_true (np.ndarray): True abundances, shape (n_samples, 3)
        ab_pred (np.ndarray): Predicted abundances, shape (n_samples, 3)
        npv_bestfit (bool): If True, use bestfit for NPV predictions
        save_path (str): If provided, save the plot to this path as a SVG
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
    if save_path:plt.savefig(save_path, format='svg')
    plt.show()

