import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


MIN_WAVELENGTH = 900
MAX_WAVELENGTH = 1690
RANDOM_STATE = 42
NPV_COLUMN = 1


# =================== DATASET PREP ======================

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


def prepare_ds_two_sets(training_file: str = 'simpler_data.csv',
                        testing_file: str = 'simpler_data_atmo.csv') -> np.ndarray:
    """Return the normalized dataset in  <file> as an array, split with test size <test size>.

    Note: Normalization to mean 0 and standard deviation 1 is done based on the mean and st. dev.
    of the training data only to avoid data leakage, but is applied to both training and testing.

    The returns are all numpy arrays with the following shapes:
    x_train: (1378, 81)
    x_test: (345, 81)
    y_train: (1378, 3)
    y_test: (345, 3)
    """

    training_ds = pd.read_csv(training_file)
    wavelength_cols = [col for col in training_ds.columns if
                       col.isdigit() and MIN_WAVELENGTH <= int(col) <= MAX_WAVELENGTH]
    target_cols = ['gv_fraction', 'npv_fraction', 'soil_fraction']
    x_train = training_ds[wavelength_cols]
    y_train = training_ds[target_cols]
    testing_ds = pd.read_csv(testing_file)
    x_test = testing_ds[wavelength_cols]
    y_test = testing_ds[target_cols]
    mean, sd = x_train.mean(axis=0), x_train.std(axis=0)
    x_train, x_test = (x_train - mean) / sd, (x_test - mean) / sd
    return x_train.to_numpy(), x_test.to_numpy(), y_train.to_numpy(), y_test.to_numpy()


def prepare_ds_rowwise(file: str = 'simpler_data.csv', test_size: float = 0.2) -> np.ndarray:
    """Return the normalized dataset in  <file> as an array, split with test size <test size>.

    Note: Normalization is done per-row (per-sample) to mean 0 and standard deviation 1,
    applied independently to both training and testing rows.

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
    x = ds[wavelength_cols].to_numpy()
    y = ds[target_cols].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=RANDOM_STATE)

    # Normalize each row to mean 0, std 1
    def row_normalize(arr: np.ndarray) -> np.ndarray:
        mean = arr.mean(axis=1, keepdims=True)  # (n, 1)
        sd = arr.std(axis=1, keepdims=True)  # (n, 1)
        sd = np.where(sd == 0, 1, sd)  # avoid division by zero
        return (arr - mean) / sd

    x_train = row_normalize(x_train)
    x_test = row_normalize(x_test)

    return x_train, x_test, y_train, y_test


# =================== DEFINE MODEL ======================


def define_mvlr_model(wmu: float, bmu: float, wsig: float, bsig: float, x_train: np.ndarray,
                      y_train: np.ndarray) -> az.InferenceData:
    """ Return the az.InferenceData (inlcluding the posterior) and pm.Model of the following
    statistical model:

    Let a be the 3 predicted abundances,
        a ~ N(w*x + b, sigma)
        w_1;1, w_1;2. ... w_1;81, w_2;1, w_2;2,...w_2;81, w_3;1, w_3;2, ...W_3;81 ~ N(wmu, wsig)
        b_1;1, b_1;2. ... b_1;81, b_2;1, b_2;2,...b_2;81, b_3;1, b_3;2, ...b_3;81 ~ N(bmu, bsig)
        sigma ~ HN(0.1)

        Preconditions: wsig, bsig > 0, file is in the format of simpler_data.csv
    """
    l = x_train.shape[1]
    with pm.Model() as model:
        # priors
        w = pm.Normal("w", mu=wmu, sigma=wsig, shape=(l, 3))
        b = pm.Normal("b", mu=bmu, sigma=bsig, shape=(3,))
        sigma = pm.HalfNormal("sigma", sigma=0.1, shape=(3,))
        # linear predictor
        a_linear = pm.math.dot(x_train, w) + b
        A_obs = pm.Normal("A_obs", mu=a_linear, sigma=sigma, observed=y_train)
        trace = pm.sample(
            1000, tune=1000, chains=4,
            target_accept=0.95,
            max_treedepth=15,
            return_inferencedata=True,
            progressbar=True
        )
    return trace, model


def define_dirichlet_model(wmu: float, bmu: float, wsig: float, bsig: float, x_train: np.ndarray,
                           y_train: np.ndarray) -> az.InferenceData:
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


# =================== PLOT DIAGNOSTICS ======================
def plot_diagnostics(trace: az.InferenceData, save_path: str = None) -> None:
    """Plot the convergence diagnostics for <trace>"""
    mpl.rcParams['font.family'] = 'Times New Roman'
    print("\n----- Convergence diagnostics -----")
    summary_df = az.summary(trace, round_to=3)
    if save_path:
        summary_df.to_csv(save_path + '/convergence_diagnostics.csv')
    az.plot_trace(trace)  # plot trace diagnostics
    plt.show(block=False)
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


# =================== MAKE PREDICTIONS ======================

def make_predictions_mvlr(trace: az.InferenceData, x_test: np.ndarray) -> np.ndarray:
    """Return posterior predictive samples as a NumPy array, including aleatoric uncertainty."""
    # Extract posterior
    w_post = trace.posterior["w"]
    b_post = trace.posterior["b"]
    sigma_post = trace.posterior["sigma"]

    # Stack chains + draws
    w_samples = (
        w_post
        .stack(draws=("chain", "draw"))
        .transpose("draws", "w_dim_0", "w_dim_1")
        .values
    )  # (n_draws, 81, 3)
    b_samples = (
        b_post
        .stack(draws=("chain", "draw"))
        .transpose("draws", "b_dim_0")
        .values
    )  # (n_draws, 3)
    sigma_samples = (
        sigma_post
        .stack(draws=("chain", "draw"))
        .transpose("draws", "sigma_dim_0")
        .values
    )  # (n_draws, 3)

    n_draws = w_samples.shape[0]
    n_samples = x_test.shape[0]
    pred_samples = np.empty((n_draws, n_samples, 3))

    for i in range(n_draws):
        linear = x_test @ w_samples[i] + b_samples[i]  # (n_samples, 3)

        # Add aleatoric noise using the posterior sigma for this draw
        noise = np.random.normal(loc=0, scale=sigma_samples[i], size=linear.shape)
        linear = linear + noise  # (n_samples, 3)

        linear = np.clip(linear, 0, None)
        row_sums = linear.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        pred_samples[i] = linear / row_sums

    return pred_samples


def make_predictions_mvlr_epistemic(trace: az.InferenceData, x_test: np.ndarray) -> np.ndarray:
    """Return posterior predictive samples as a NumPy array, epistemic uncertainty only."""
    # Extract posterior
    w_post = trace.posterior["w"]
    b_post = trace.posterior["b"]

    # Stack chains + draws
    w_samples = (
        w_post
        .stack(draws=("chain", "draw"))
        .transpose("draws", "w_dim_0", "w_dim_1")
        .values
    )  # (n_draws, 81, 3)
    b_samples = (
        b_post
        .stack(draws=("chain", "draw"))
        .transpose("draws", "b_dim_0")
        .values
    )  # (n_draws, 3)

    n_draws = w_samples.shape[0]
    n_samples = x_test.shape[0]
    pred_samples = np.empty((n_draws, n_samples, 3))

    for i in range(n_draws):
        linear = x_test @ w_samples[i] + b_samples[i]  # (n_samples, 3)

        # Clip and normalise to simplex — no aleatoric noise added
        linear = np.clip(linear, 0, None)
        row_sums = linear.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        pred_samples[i] = linear / row_sums

    return pred_samples


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
    w = trace.posterior["w"].stack(sample=("chain", "draw")).values
    b = trace.posterior["b"].stack(sample=("chain", "draw")).values
    print(f"w shape: {w.shape}, b shape: {b.shape}")
    s = w.shape[-1]
    n = x_test.shape[0]
    k = b.shape[0]

    linear = np.einsum("nl,lks->snk", x_test, w) + b.T[:, None, :]
    print(
        f"linear shape: {linear.shape}, any nan: {np.isnan(linear).any()}, any inf: {np.isinf(linear).any()}")

    alpha = np.log1p(np.exp(linear)) + 1e-6
    print(f"alpha shape: {alpha.shape}, min: {alpha.min():.4f}, max: {alpha.max():.4f}")

    rng = np.random.default_rng()
    pred_samples = np.array([
        rng.dirichlet(a) for a in alpha.reshape(-1, k)
    ]).reshape(s, n, k)

    return pred_samples


def make_predictions_dirichlet_model_epistemic(trace: az.InferenceData,
                                               x_test: np.ndarray) -> np.ndarray:
    """Return posterior predictive samples as a NumPy array. Predictions are made considering
    epistemic uncertainty only.

    Output shape:
        (S, N, K = 3)

    where:
        S = total posterior draws
        N = number of test samples
        K = 3 = number of Dirichlet components
    """
    # Get posterior samples
    w = trace.posterior["w"].stack(sample=("chain", "draw")).values  # W shape: (81, K, S)
    b = trace.posterior["b"].stack(sample=("chain", "draw")).values  # b shape: (K, S)
    print(w.shape, b.shape)
    s = w.shape[-1]  # total posterior draws
    n = x_test.shape[0]  # number of test points
    k = b.shape[0]  # number of dirichlet components
    # vectorized linear predictor
    try:
        linear = np.einsum("nl,lks->snk", x_test, w) + b.T[:, None, :]  # x_n*w + b
        alpha = np.log1p(np.exp(linear)) + 1e-6  # shape(S,N,K)
        pred_samples = alpha / alpha.sum(axis=2, keepdims=True)
        return pred_samples
    except Exception:
        print('Could not perform linear prediction.')


# =================== EVALUATE MODEL HELPER ======================

def evaluate_model(predicted_value: np.ndarray, true_value: np.ndarray, p: bool = True) \
        -> tuple[float]:
    """Return (and print if <p>) the rmse, r2 and r2 of y = x of <true_value> vs <predicted_value>
    """
    rmse = np.sqrt(np.mean((true_value - predicted_value) ** 2))
    rmse_npv = np.sqrt(np.mean((true_value[:, NPV_COLUMN] - predicted_value[:, NPV_COLUMN]) ** 2))
    r2_npv = r2_score(true_value[:, NPV_COLUMN], predicted_value[:, NPV_COLUMN])
    r2 = r2_score(true_value, predicted_value)
    ss_res = np.sum((predicted_value - true_value) ** 2)
    ss_tot = np.sum((true_value - true_value.mean(axis=0)) ** 2)  # per-column mean
    r2_y_eq_x = 1 - ss_res / ss_tot

    ss_res = np.sum((predicted_value[:, NPV_COLUMN] - true_value[:, NPV_COLUMN]) ** 2)
    ss_tot = np.sum((true_value[:, NPV_COLUMN] - true_value[:, NPV_COLUMN].mean()) ** 2)
    r2_y_eq_x_npv = 1 - ss_res / ss_tot

    if p:
        print(f"RMSE: {rmse:.4f}")
        print(f"RMSE (NPV): {rmse_npv:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"R² (NPV): {r2_npv:.4f}")
        print(f"R² (y = x): {r2_y_eq_x:.4f}")
        print(f"R² (y = x) (NPV): {r2_y_eq_x_npv:.4f}")
    return (rmse, rmse_npv, r2_npv, r2, r2_y_eq_x, r2_y_eq_x_npv)

# =================== PLOT MODEL HELPER ======================


def plot_preds(ab_true: np.ndarray, ab_pred: np.ndarray, npv_bestfit: bool = True,
               save_path: str = None, title: str = None) -> None:
    """Plot predicted vs true abundances for three classes. Optionally bestfits for the npv.

    Args:
        ab_true (np.ndarray): True abundances, shape (n_samples, 3)
        ab_pred (np.ndarray): Predicted abundances, shape (n_samples, 3)
        npv_bestfit (bool): If True, use bestfit for NPV predictions
        save_path (str): If provided, save the plot to this path as a SVG
        save_path (str): If provided, make it the title of the graph
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
    if title:
        plt.title(title)
    else:
        plt.title("Predicted vs True Abundances")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='svg')
    plt.show(block=False)


def plot_pred_hdpis(true_abundances: np.ndarray, pred_samples: np.ndarray,
                    prob=0.90, save_path: str = None, title: str = None) -> None:
    """
    pred_samples shape: (S, N, K)
    true_abundances shape: (N, K)
    """
    mpl.rcParams['font.family'] = 'Times New Roman'
    colors = ['green', 'orange', 'saddlebrown']
    labels = ['GV', 'NPV', 'Soil']
    markers = ['o', '^', 's']
    pred_samples = np.asarray(pred_samples)
    if pred_samples.ndim != 3:
        raise ValueError(
            f"Expected shape (S, N, K). Got {pred_samples.shape}"
        )
    s, n, k = pred_samples.shape

    central_preds = np.median(pred_samples, axis=0)  # (n, k)

    # HDPI computation (axis=0)
    interval_size = int(np.floor(prob * s))
    sorted_samples = np.sort(pred_samples, axis=0)
    widths = (sorted_samples[interval_size:, :, :] - sorted_samples[:s - interval_size, :, :])
    min_idx = np.argmin(widths, axis=0)
    lower = np.empty((n, k))
    upper = np.empty((n, k))
    for ni in range(n):
        for ki in range(k):
            idx = min_idx[ni, ki]
            lower[ni, ki] = sorted_samples[idx, ni, ki]
            upper[ni, ki] = sorted_samples[idx + interval_size, ni, ki]

    # Compute offsets, clipping to 0 as floating point safety net
    lower_err = np.clip(central_preds - lower, 0, None)
    upper_err = np.clip(upper - central_preds, 0, None)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    for ki in range(k):
        ax.scatter(
            true_abundances[:, ki],
            central_preds[:, ki],
            color=colors[ki],
            marker=markers[ki],
            alpha=0.6,
            edgecolor='k',
            label=labels[ki]
        )
        ax.errorbar(
            true_abundances[:, ki],
            central_preds[:, ki],
            yerr=[lower_err[:, ki], upper_err[:, ki]],
            fmt='none',
            ecolor=colors[ki],
            alpha=0.5
        )

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("True Abundance")
    ax.set_ylabel("Predicted Abundance")
    if title:
        ax.set_title(title)
    else:
        ax.set_title('True vs Predicted Abundance using posterior median with 90% HDPI')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + ".svg", format="svg")
    plt.show(block=False)


# =================== PLOT AND EVALUATE MODEL  ======================

def evaluate_and_graph_mvlr_model(pred_samples: np.ndarray,
                                  true_abundances: np.ndarray, save_path: str = None) -> None:
    """Print the prediction accuracy and plot the predictions using the mean and median of the
    predicted abundances, and print and plot the HDPIs vs. true abundances.
    """
    print("\n----- Prediction results using the mean of the predictions -----")
    mean_preds = pred_samples.mean(axis=0)
    evaluate_model(mean_preds, true_abundances)
    plot_preds(true_abundances, mean_preds,
               save_path=(save_path + '/mean_predictions_mvlr.svg') if save_path else None,
               title="Predicted vs True Abundances using the mean of the predictive posterior")

    print("\n----- Plotting epistemic uncertainty (median ± 90% HDPI) -----")
    plot_pred_hdpis(true_abundances, pred_samples,
                    save_path=(save_path + '/hdpis_mvlr_plot') if save_path else None,
                    title="Predicted vs True Abundances using the median of the predictive "
                          "posterior ± 90% HDPI")


def evaluate_and_graph_dirichlet_model(pred_samples: np.ndarray,
                                       true_abundances: np.ndarray, save_path: str = None) -> None:
    """Print the prediction accuracy and plot the predictions using the mean and median of the
            predicted abundances, and print and plot the HDPIs vs. true abundances.
    """
    print("\n----- Prediction results using the mean of the predictions -----")
    mean_preds = pred_samples.mean(axis=0)
    evaluate_model(mean_preds, true_abundances)
    plot_preds(true_abundances, mean_preds,
               save_path=(save_path + '/mean_predictions_dirichlet.svg') if save_path else None,
               title="Predicted vs True Abundances using the mean of the predictive posterior")

    print("\n----- Plotting epistemic uncertainty (median ± 90% HDPI) -----")
    plot_pred_hdpis(true_abundances, pred_samples,
                    save_path=(save_path + '/hdpis_dirichlet_plot') if save_path else None,
                    title="Predicted vs True Abundances using the median of the predictive "
                          "posterior ± 90% HDPI")


# =================== RUN MODEL ======================


def run_bayesian_mvlr(file: str = 'simpler_data.csv', test_size: float = 0.2, wmu: float = 0,
                      bmu: float = 0, wsig: float = 1, bsig: float = 1, save_path: str = None
                      ) -> None:
    """ Run the following bayesian statistical model and print the convergence plots and model
        results:

        Let a be the 3 predicted abundances,
        a ~ N(w*x + b, sigma)
        w_1;1, w_1;2. ... w_1;81, w_2;1, w_2;2,...w_2;81, w_3;1, w_3;2, ...W_3;81 ~ N(wmu, wsig)
        b_1;1, b_1;2. ... b_1;81, b_2;1, b_2;2,...b_2;81, b_3;1, b_3;2, ...b_3;81 ~ N(bmu, bsig)
        sigma ~ HN(0.1)

        Preconditions: wsig, bsig > 0, file is in the format of simpler_data.csv
    """
    x_train, x_test, y_train, y_test = prepare_ds(file, test_size)
    trace, model = define_mvlr_model(wmu, bmu, wsig, bsig, x_train, y_train)
    plot_diagnostics(trace, save_path=save_path)
    pred_samples = make_predictions_mvlr(trace, x_test)
    evaluate_and_graph_mvlr_model(pred_samples, y_test, save_path)


def run_bayesian_mvlr_epistemic(file: str = 'simpler_data.csv', test_size: float = 0.2,
                                wmu: float = 0,
                                bmu: float = 0, wsig: float = 1, bsig: float = 1,
                                save_path: str = None
                                ) -> None:
    """ Run the following bayesian statistical model and print the convergence plots and model
        results:

        Let a be the 3 predicted abundances,
        a ~ N(w*x + b, sigma)
        w_1;1, w_1;2. ... w_1;81, w_2;1, w_2;2,...w_2;81, w_3;1, w_3;2, ...W_3;81 ~ N(wmu, wsig)
        b_1;1, b_1;2. ... b_1;81, b_2;1, b_2;2,...b_2;81, b_3;1, b_3;2, ...b_3;81 ~ N(bmu, bsig)
        sigma ~ HN(0.1)

        Preconditions: wsig, bsig > 0, file is in the format of simpler_data.csv
    """
    x_train, x_test, y_train, y_test = prepare_ds(file, test_size)
    trace, model = define_mvlr_model(wmu, bmu, wsig, bsig, x_train, y_train)
    plot_diagnostics(trace, save_path=save_path)
    pred_samples = make_predictions_mvlr_epistemic(trace, x_test)
    evaluate_and_graph_mvlr_model(pred_samples, y_test, save_path)


def run_bayesian_mvlr_rowwise(file: str = 'simpler_data.csv', test_size: float = 0.2,
                              wmu: float = 0,
                              bmu: float = 0, wsig: float = 1, bsig: float = 1,
                              save_path: str = None
                              ) -> None:
    """ Run the following bayesian statistical model and print the convergence plots and model
        results:

        Let a be the 3 predicted abundances,
        a ~ N(w*x + b, sigma)
        w_1;1, w_1;2. ... w_1;81, w_2;1, w_2;2,...w_2;81, w_3;1, w_3;2, ...W_3;81 ~ N(wmu, wsig)
        b_1;1, b_1;2. ... b_1;81, b_2;1, b_2;2,...b_2;81, b_3;1, b_3;2, ...b_3;81 ~ N(bmu, bsig)
        sigma ~ HN(0.1)

        Preconditions: wsig, bsig > 0, file is in the format of simpler_data.csv
    """
    x_train, x_test, y_train, y_test = prepare_ds_rowwise(file, test_size)
    trace, model = define_mvlr_model(wmu, bmu, wsig, bsig, x_train, y_train)
    plot_diagnostics(trace, save_path=save_path)
    pred_samples = make_predictions_mvlr(trace, x_test)
    evaluate_and_graph_mvlr_model(pred_samples, y_test, save_path)


def run_dirichlet_model(file: str = 'simpler_data.csv', test_size: float = 0.2, wmu: float = 0,
                        bmu: float = 0, wsig: float = 1, bsig: float = 1, save_path: str = None
                        ) -> None:
    """ Run the following bayesian statistical model and print the convergence plots and model
    results:

    Let a be the 3 predicted abundances,
    a ~ Dir(softplus(w*x + b))
    w_1;1, w_1;2. ... w_1;81, w_2;1, w_2;2,...w_2;81, w_3;1, w_3;2, ...W_3;81 ~ N(wmu, wsig)
    b_1;1, b_1;2. ... b_1;81, b_2;1, b_2;2,...b_2;81, b_3;1, b_3;2, ...b_3;81 ~ N(bmu, bsig)

    Preconditions: wsig, bsig > 0, file is in the format of simpler_data.csv

    Note: Abundances == 0 are set to eps = 10^(-6) and abundances == 1 are set to 1 - eps so they
    can be fed into the model
    """
    x_train, x_test, y_train, y_test = prepare_ds(file, test_size)
    trace, model = define_dirichlet_model(wmu, bmu, wsig, bsig, x_train, y_train)
    plot_diagnostics(trace, save_path=save_path)
    pred_samples = make_predictions_dirichlet_model(trace, x_test)
    evaluate_and_graph_dirichlet_model(pred_samples, y_test, save_path)


def run_dirichlet_model_epistemic(file: str = 'simpler_data.csv', test_size: float = 0.2,
                                  wmu: float = 0,
                                  bmu: float = 0, wsig: float = 1, bsig: float = 1,
                                  save_path: str = None
                                  ) -> None:
    """ Run the following bayesian statistical model and print the convergence plots and model
    results:

    Let a be the 3 predicted abundances,
    a ~ Dir(softplus(w*x + b))
    w_1;1, w_1;2. ... w_1;81, w_2;1, w_2;2,...w_2;81, w_3;1, w_3;2, ...W_3;81 ~ N(wmu, wsig)
    b_1;1, b_1;2. ... b_1;81, b_2;1, b_2;2,...b_2;81, b_3;1, b_3;2, ...b_3;81 ~ N(bmu, bsig)

    Make predictions by considering epistemic uncertainty only.

    Preconditions: wsig, bsig > 0, file is in the format of simpler_data.csv

    Note: Abundances == 0 are set to eps = 10^(-6) and abundances == 1 are set to 1 - eps so they
    can be fed into the model
    """
    x_train, x_test, y_train, y_test = prepare_ds(file, test_size)
    trace, model = define_dirichlet_model(wmu, bmu, wsig, bsig, x_train, y_train)
    plot_diagnostics(trace, save_path=save_path)
    pred_samples = make_predictions_dirichlet_model_epistemic(trace, x_test)
    evaluate_and_graph_dirichlet_model(pred_samples, y_test, save_path)


def run_dirichlet_model_rowwise(file: str = 'simpler_data.csv', test_size: float = 0.2,
                                wmu: float = 0,
                                bmu: float = 0, wsig: float = 1, bsig: float = 1,
                                save_path: str = None
                                ) -> None:
    """ Run the following bayesian statistical model and print the convergence plots and model
    results:

    Let a be the 3 predicted abundances,
    a ~ Dir(softplus(w*x + b))
    w_1;1, w_1;2. ... w_1;81, w_2;1, w_2;2,...w_2;81, w_3;1, w_3;2, ...W_3;81 ~ N(wmu, wsig)
    b_1;1, b_1;2. ... b_1;81, b_2;1, b_2;2,...b_2;81, b_3;1, b_3;2, ...b_3;81 ~ N(bmu, bsig)

    Normalize data rowwise.

    Preconditions: wsig, bsig > 0, file is in the format of simpler_data.csv

    Note: Abundances == 0 are set to eps = 10^(-6) and abundances == 1 are set to 1 - eps so they
    can be fed into the model
    """
    x_train, x_test, y_train, y_test = prepare_ds(file, test_size)
    trace, model = define_dirichlet_model(wmu, bmu, wsig, bsig, x_train, y_train)
    plot_diagnostics(trace, save_path=save_path)
    pred_samples = make_predictions_dirichlet_model(trace, x_test)
    evaluate_and_graph_dirichlet_model(pred_samples, y_test, save_path)


def run_dirichlet_model_two_datasets(training_file: str = 'simpler_data.csv',
                                     testing_file: str = 'simpler_data_atmo.csv', wmu: float = 0,
                                     bmu: float = 0, wsig: float = 1, bsig: float = 1,
                                     save_path: str = None
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
    x_train, x_test, y_train, y_test = prepare_ds_two_sets(training_file, testing_file)
    trace, model = define_dirichlet_model(wmu, bmu, wsig, bsig, x_train, y_train)
    plot_diagnostics(trace, save_path=save_path)
    pred_samples = make_predictions_dirichlet_model(trace, x_test)
    evaluate_and_graph_dirichlet_model(pred_samples, y_test, save_path)
