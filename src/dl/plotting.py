import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

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

def plot_avg_losses(epochs:(np.ndarray), losses:(np.ndarray), save_path:(str), model_name:(str)):
    """
    Plots the average training and validation losses

    Args:
        epochs (np.ndarray): A numpy array of epochs
        losses (np.ndarray): A numpy array of losses wher 0th column is training and 1st is validation
        save_path (str): Save the plot to this path as a SVG
        model_name (str): The name of the model that was used
    """
    mpl.rcParams['font.family'] = 'Times New Roman'
    colors = ['#1f77b4', '#ff7f0e']
    labels = ['Training', 'Validation']

    # Start plot
    plt.figure(figsize=(8,6))

    # The plot itself
    for i, (label, color) in enumerate(zip(labels, colors)):
        plt.plot(
            epochs, 
            losses[:,i],
            label=label,
            color=color
        )
    
    # Finish the plot
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title(f"{model_name}, Average Losses per Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.show()

def plot_train_losses(steps:(np.ndarray), losses:(np.ndarray), loss_names:(list[str]), save_path:(str), model_name:(str)):
    """
    Plots the training losses per steps

    Args:
        steps (np.ndarray): The numpy array of steps
        losses (np.ndarray): A numpy array of all loss components in each row
        loss_names (list[str]): A list of loss names for total and components
        save_path (str): Save the plot to this path as a SVG
        model_name (str): The name of the model that was used
    """
    mpl.rcParams['font.family'] = 'Times New Roman'
    colors = [
        '#000000', 
        '#E69F00', 
        '#56B4E9', 
        '#009E73', 
        '#F0E442', 
        '#0072B2', 
        '#D55E00'  
    ]

    # Start plot
    plt.figure(figsize=(8,6))

    # The plot itself
    for i, (label, color) in enumerate(zip(loss_names, colors)):
        plt.plot(
            steps,
            losses[:,i],
            label=label,
            color=color
        )
    
    # Finish the plot
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(f"{model_name}, Loss Components per Steps")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.show()

def plot_metrics(epochs: (np.ndarray), metrics:(np.ndarray), metric_names:(list[str]), save_path:(str), model_name:(str)):
    """
    Plots the metrics throughout training

    Args:
        epochs (np.ndarray): The numpy array of epochs
        metrics (np.ndarray): The numpy array of different metrics, rows as epochs cols as type
        metric_names (list[str]): The list of names of the metrics
        save_path (str): Saving path
        model_name (str): The name of the model
    """
    mpl.rcParams['font.family'] = 'Times New Roman'
    colors = [
        '#000000', 
        '#E69F00', 
        '#56B4E9', 
        '#009E73', 
        '#F0E442', 
        '#0072B2', 
        '#D55E00'  
    ]

    # Start plot
    plt.figure(figsize=(8,6))

    # The plot itself
    for i, (label, color) in enumerate(zip(metric_names, colors)):
        plt.plot(
            epochs,
            metrics[:,i],
            label=label,
            color=color
        )

    # Finish the plot
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.ylim(top=1.2, bottom=-5)
    plt.title(f"{model_name}, Metrics per Validation in Epochs")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.show()

def pareto_plot(x_axis:(np.ndarray), x_label:(str), y_axis:(np.ndarray), y_component_label:(list[str]), y_label:(str), title:(str), save_path:(str)):
    """
    Pareto front plot for anything

    Args:
        x_axis (np.ndarray): A 2D numpy array of the x axis, columns represent different models
        x_label (str): The name of xlabel
        y_axis (np.ndarray): A 2D numpy array of the x axis, columns represent different models
        y_component_label (list[str]): A list for the label of plots
        y_label (str): The name of ylabel
        title (str): The full title as a str
        save_path (str): The save path
    """
    mpl.rcParams['font.family'] = 'Times New Roman'
    colors = [
        '#E69F00', 
        '#56B4E9', 
        '#009E73', 
        '#F0E442', 
        '#0072B2', 
        '#D55E00'  
    ]

    # Start the plot
    plt.figure(figsize=(8,6))

    # The plot itself 
    for i, (component_label, color) in enumerate(zip(y_component_label, colors)):

        # Extract columns
        xi = x_axis[:, i]
        yi = y_axis[:, i]

        # Filter out NaNs and sort by x_axis values
        mask = ~np.isnan(xi) & ~np.isnan(yi)
        xi, yi = xi[mask], yi[mask]
        
        sort_idx = np.argsort(xi)
        xi_sorted = xi[sort_idx]
        yi_sorted = yi[sort_idx]

        plt.plot(
            xi_sorted,
            yi_sorted,
            label=component_label,
            color=color,
            linestyle='-', linewidth=2, alpha=0.7
        )

        plt.scatter(
            xi_sorted,
            yi_sorted,
            color=color,
            edgecolors='black',
            zorder=3
        )
 
    # Finish the plot
    plt.xlabel(x_label)
    plt.xscale('log')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()
