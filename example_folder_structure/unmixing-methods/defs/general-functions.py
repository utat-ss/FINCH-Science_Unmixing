import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


MIN_WAVELENGTH = 900
MAX_WAVELENGTH = 1700
RANDOM_STATE = 42
NPV_COLUMN = 1


def prepare_ds(file: str = 'simpler_data.csv', test_size: float = 0.2) -> np.ndarray:
    """Return the normalized dataset in  <file> as an array, split with test size <test size>.

    Note: Normalization to mean 0 and standard deviation 1 is done based on the mean and st. dev.
    of the training data only to avoid data leakage, but is applied to both training and testing.

    The returns are all numpy arrays with the following shapes:
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


def evaluate_model(predicted_value: np.ndarray, true_value: np.ndarray, p: bool = True) -> tuple[float]:
    """Return (and print if <p>) the rmse, r2 and r2 of y = x of <true_value> vs <predicted_value>
    """
    rmse = np.sqrt(np.mean((true_value - predicted_value) ** 2))
    rmse_npv = np.sqrt(np.mean((true_value[:, NPV_COLUMN] - predicted_value[:, NPV_COLUMN]) ** 2))
    r2_npv = r2_score(true_value[:, NPV_COLUMN], predicted_value[:, NPV_COLUMN])
    r2 = r2_score(true_value, predicted_value)
    ss_res = np.sum((predicted_value - true_value) ** 2)
    ss_tot = np.sum((true_value - np.mean(true_value)) ** 2)
    r2_y_eq_x = 1 - ss_res / ss_tot
    ss_res = np.sum((predicted_value[:, NPV_COLUMN] - true_value[:, NPV_COLUMN]) ** 2)
    ss_tot = np.sum((true_value[:, NPV_COLUMN] -
                     np.mean(true_value[:, NPV_COLUMN])) ** 2)
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
               save_path: str = None, title: str = 'Predicted vs True Abundances') -> None:
    """Plot predicted vs true abundances for three classes. Optionally bestfits for the npv.

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
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='svg')
    plt.show(block=False)
    print('plot_preds_done')


def run_dirichlet_model(file: str = 'simpler_data.csv', test_size: float = 0.2, wmu: float = 0,
                        bmu: float = 0, wsig: float = 1, bsig: float = 1, save_path: str = None
                        ) -> None:
    """ Run the following bayesian statistical model and print the convergence plots and model
    results:

    Let a be the 3 predicted abundances,
    a ~ Dir(softplus(w*x + b)
    w_1;1, w_1;2. ... w_1;81, w_2;1, w_2;2,...w_2;81, w_3;1, w_3;2, ...W_3;81 ~ N(wmu, wsig)
    b_1;1, b_1;2. ... b_1;81, b_2;1, b_2;2,...b_2;81, b_3;1, b_3;2, ...b_3;81 ~ N(bmu, bsig)

    Preconditions: wsig, bsig > 0, file is in the format of simpler_data.csv

    Note: Abundances == 0 are set to eps = 10^(-6) and abundances == 1 are set to 1 - eps so they
    can be fed into the model
    """
    x_train, x_test, y_train, y_test = prepare_ds(file, test_size)
    trace, model = define_dirichlet_model(wmu, bmu, wsig, bsig, x_train, y_train)
    print('trace acquired')
    plot_diagnostics(trace, save_path=save_path)
    print('diagnostics plotted')
    pred_samples = make_predictions_dirichlet_model(trace, x_test)
    print('predictions made')
    evaluate_and_graph_dirichlet_model(pred_samples, y_test, save_path)
    print('model evaluated')


def make_predictions_dirichlet_model(trace: az.InferenceData,
                                     x_test: np.ndarray) -> np.ndarray:
    """Return posterior predictive samples as a NumPy array.

    Output shape:
        (S, N, K = 3)

    where:
        S = total posterior draws
        N = number of test samples
        K = 3 = number of Dirichlet components
    """
    print('make predictions started')
    # Get posterior samples
    w = trace.posterior["w"].stack(sample=("chain", "draw")).values  # W shape: (81, K, S)
    b = trace.posterior["b"].stack(sample=("chain", "draw")).values  # b shape: (K, S)
    print('posterior samples acquired')
    print(w.shape, b.shape)
    s = w.shape[-1]  # total posterior draws
    n = x_test.shape[0]  # number of test points
    k = b.shape[0]  # number of dirichlet components
    # vectorized linear predictor
    print('vectorized linear predictor')
    try:
        linear = np.einsum("nl,lks->snk", x_test, w) + b.T[:, None, :]  # x_n*w + b
        alpha = np.log1p(np.exp(linear)) + 1e-6  # shape(S,N,K)
        pred_samples = alpha / alpha.sum(axis=2, keepdims=True)
        return pred_samples
    except Exception:
        print('Could not perform linear prediction.')


def evaluate_and_graph_dirichlet_model(pred_samples: np.ndarray,
                                       true_abundances: np.ndarray, save_path: str = None) -> None:
    """Print the prediction accuracy and plot the predictions using the mean and median of the
    predicted abundances, and print and plot the HDPIs vs. true abundances.
    """
    # evaluate prediction accuracy of the mean of each predictive posterior
    print("\n----- Prediction results using the mean of the predictions -----")
    mean_preds = pred_samples.mean(axis=0)
    evaluate_model(mean_preds, true_abundances)
    plot_preds(mean_preds, true_abundances, save_path=(save_path + '/mean_predictions.svg'),
               title='Predicted vs True Abundances for Dirichlet Model using Prediction Means')
    # evaluate prediction accuracy of the median of each predictive posterior
    print("\n----- Prediction results using the median of the predictions -----")
    median_preds = np.median(pred_samples, axis=0)
    evaluate_model(median_preds, true_abundances)
    plot_preds(median_preds, true_abundances, save_path=(save_path + '/median_predictions.svg'),
               title='Predicted vs True Abundances for Dirichlet Model using Prediction Medians')


def plot_diagnostics(trace: az.InferenceData, save_path: str = None) -> None:
    """Plot the convergence diagnostics for <trace>"""
    mpl.rcParams['font.family'] = 'Times New Roman'
    print("\n----- Convergence diagnostics -----")
    summary_df = az.summary(trace, round_to=3)
    if save_path:
        summary_df.to_csv(save_path + '/convergence_diagnostics.csv')
    az.plot_trace(trace)  # plot trace diagnostics
    summary_df["rel_mcse"] = summary_df["mcse_mean"] / summary_df["sd"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    # Effective sample size
    axs[0].hist(summary_df["ess_bulk"], bins=30, color='saddlebrown', edgecolor='black')
    axs[0].set_title("Effective Sample Size (Bulk) Frequency")
    axs[0].set_xlabel("Effective Sample Size")
    # Rhat statistic
    axs[1].hist(summary_df["r_hat"], bins=30, color='saddlebrown', edgecolor='black')
    axs[1].set_title("R-hat Statistics Frequency")
    axs[1].set_xlabel("R-hat Statistic")
    # Relative MCSE
    axs[2].hist(summary_df["rel_mcse"], bins=30, color="saddlebrown", edgecolor="black")
    axs[2].axvline(0.05, color="red", linestyle="--", label="MCSE/SD = 5%")
    axs[2].set_title("Relative Monte Carlo Standard Error Frequency")
    axs[2].set_xlabel("Relative Monte Carlo Standard Error")
    axs[2].legend()
    for ax in axs:
        ax.set_ylabel("Number of Model Parameter")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + '/diagnostic_plots.svg', format='svg')
    plt.show(block=False)
    plt.close()
    print('plot_diagnostics done')



def define_dirichlet_model(wmu: float, bmu: float, wsig: float, bsig: float, x_train: float,
                           y_train: float) -> az.InferenceData:
    """ Return the az.InferenceData (inlcluding the posterior) and pm.Model of the following
    statistical model:

    Let a be the 3 predicted abundances,
    a ~ Dir(softplus(w*x + b)
    w_1;1, w_1;2. ... w_1;81, w_2;1, w_2;2,...w_2;81, w_3;1, w_3;2, ...W_3;81 ~ N(wmu, wsig)
    b_1;1, b_1;2. ... b_1;81, b_2;1, b_2;2,...b_2;81, b_3;1, b_3;2, ...b_3;81 ~ N(bmu, bsig)

   <wmu>, <wsig>, <bmu>, <bsig> are the priors, <x_train> the predictor variables and <y_train>
   their correspondent target variables

    Preconditions: wsig, bsig > 0

    Note: Abundances == 0 are set to eps = 10^(-6) and abundances == 1 are set to 1 - eps so they
    can be fed into the model
    """
    eps = 1e-6
    y_clipped = np.clip(y_train, eps, 1 - eps)
    l = x_train.shape[1]
    with pm.Model() as model:
        # priors
        w = pm.Normal("w", mu=wmu, sigma=wsig, shape=(l, 3))
        b = pm.Normal("b", mu=bmu, sigma=bsig, shape=(3,))
        # linear predictor
        a_linear = pm.math.dot(x_train, w) + b
        # softplus to make alpha positive
        alpha = pm.math.log1pexp(a_linear) + eps
        # dirichlet likelihood
        a_obs = pm.Dirichlet(
            "A_obs",
            a=alpha,
            observed=y_clipped
        )
        trace = pm.sample(
            1000, tune=1000, chains=4,
            target_accept=0.95,
            max_treedepth=15,
            return_inferencedata=True,
            progressbar=True
        )
    return trace, model


if __name__ == "__main__":
    run_dirichlet_model(file="simpler_data.csv")
