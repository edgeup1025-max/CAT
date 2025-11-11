import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression


class ParameterDiagnostics:
    """
    Advanced parameter diagnostics for GRM / IRT models with
    entropy, discrimination, DIF, and MLflow auto-logging for sklearn models.
    """

    def __init__(self,
                 beta: pd.DataFrame,
                 alpha: pd.DataFrame,
                 output_dir: str = None):
        if isinstance(alpha, pd.DataFrame):
            self.alphas = alpha.squeeze().to_numpy()
        elif isinstance(alpha, pd.Series):
            self.alphas = alpha.to_numpy()
        else:
            self.alphas = np.asarray(alpha)
        output_dir = r'D:\WORKSPACE\OFIICE_WORKS\model\src\saver'
        self.betas = beta.copy()
        self.beta_np = self.betas.to_numpy()
        self.n_items = self.beta_np.shape[0]
        self.n_thresholds = self.beta_np.shape[1]
        self.thetas_grid = np.linspace(-3, 3, 200)
        self.output_dir = output_dir or os.getcwd()
        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    # Helper: Log sklearn model to MLflow
    # ----------------------------------------------------------------------
    @staticmethod
    def log_sklearn_model(model,
                          model_name="sklearn_model",
                          subdir="ml_models"):
        """Log sklearn model into MLflow (if active run)."""
        try:
            import mlflow
            import mlflow.sklearn

            if mlflow.active_run() is not None:
                model_dir = os.path.join(os.getcwd(), subdir)
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"{model_name}.pkl")
                mlflow.sklearn.log_model(sk_model=model,
                                         artifact_path=model_name,
                                         registered_model_name=model_name)
                print(f"✅ MLflow model logged: {model_name}")
        except Exception as e:
            print(f"⚠️ Sklearn model logging failed for {model_name}: {e}")

    # ----------------------------------------------------------------------
    # BASIC CHECKS
    # ----------------------------------------------------------------------
    def check_order(self):
        disordered_items = [
            int(i) for i, row in self.betas.iterrows()
            if not np.all(np.diff(row.values) > 0)
        ]
        return {
            "n_disordered": len(disordered_items),
            "disordered_indices": disordered_items
        }

    # ----------------------------------------------------------------------
    # CCC PLOTS
    # ----------------------------------------------------------------------
    def plot_cccs(self, n_cols=5, save_html=True, save_png=False):
        from matplotlib import pyplot as plt

        n_items = self.n_items
        n_rows = int(np.ceil(n_items / n_cols))
        fig_mpl, axes = plt.subplots(n_rows,
                                     n_cols,
                                     figsize=(n_cols * 4, n_rows * 3),
                                     squeeze=False)

        for i in range(n_items):
            row, col = divmod(i, n_cols)
            ax = axes[row][col]
            b, a = self.beta_np[i], self.alphas[i]

            P_star = expit(a * (self.thetas_grid[:, None] - b[None, :]))
            P_full = np.concatenate([
                np.ones((len(self.thetas_grid), 1)), P_star,
                np.zeros((len(self.thetas_grid), 1))
            ],
                                    axis=1)
            P_cat = P_full[:, :-1] - P_full[:, 1:]
            for k in range(P_cat.shape[1]):
                ax.plot(self.thetas_grid, P_cat[:, k], label=f'Cat {k+1}')
            ax.set_title(f'Item {i+1} | α={a:.2f}')
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)
        plt.tight_layout()

        png_path = os.path.join(self.output_dir, "cccs_plot.png")
        fig_mpl.savefig(png_path, dpi=150)
        html_file = None

        if save_html:
            html_file = os.path.join(self.output_dir, "cccs_plot.html")
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(
                    f"<html><body><h2>Category Characteristic Curves (CCC)</h2>"
                    f"<img src='{os.path.basename(png_path)}' style='max-width:100%;height:auto;'/></body></html>"
                )
        if not save_png:
            plt.close(fig_mpl)
        return fig_mpl, html_file

    # ----------------------------------------------------------------------
    # CORE CALCULATIONS
    # ----------------------------------------------------------------------
    def compute_item_probabilities(self, thetas=None):
        thetas = self.thetas_grid if thetas is None else np.asarray(thetas)
        n_t, n_items, n_cat = len(thetas), self.n_items, self.n_thresholds + 1
        P = np.zeros((n_items, n_t, n_cat))
        for j in range(n_items):
            aj, bj = self.alphas[j], self.beta_np[j]
            P_star = expit(aj * (thetas[:, None] - bj[None, :]))
            P_full = np.concatenate(
                [np.ones((n_t, 1)), P_star,
                 np.zeros((n_t, 1))], axis=1)
            P_cat = P_full[:, :-1] - P_full[:, 1:]
            P_cat = np.clip(P_cat, 1e-9, 1.0)
            P_cat /= P_cat.sum(axis=1, keepdims=True)
            P[j] = P_cat
        return P

    def compute_entropy(self, thetas=None):
        P = self.compute_item_probabilities(thetas)
        entropy = -np.sum(P * np.log(P), axis=2)
        mean_entropy = entropy.mean(axis=1)
        median_entropy = np.median(entropy, axis=1)
        return {
            "entropy_grid": entropy,
            "mean_entropy_per_item": mean_entropy,
            "median_entropy_per_item": median_entropy,
            "overall_mean_entropy": float(mean_entropy.mean()),
        }

    def compute_item_information(self, thetas=None):
        P = self.compute_item_probabilities(thetas)
        info = np.zeros((self.n_items, len(self.thetas_grid)))
        for j in range(self.n_items):
            aj, Pj = self.alphas[j], P[j]
            info[j] = (aj**2) * np.sum(Pj * (1 - Pj), axis=1)
        return info, info.sum(axis=0)

    # ----------------------------------------------------------------------
    # DIF ANALYSIS — AUTO MLflow LOGGING
    # ----------------------------------------------------------------------
    def compute_dif(self, X, group, thetas=None):
        """
        Perform DIF analysis with LinearRegression and log each model to MLflow.
        """
        X, group = np.asarray(X), np.asarray(group)
        if thetas is None:
            thetas = np.zeros(X.shape[0])
        results = []

        for j in range(X.shape[1]):
            y = X[:, j].astype(float)
            lr = LinearRegression().fit(thetas.reshape(-1, 1), y)

            # 🔹 Log this regression model into MLflow for traceability
            self.log_sklearn_model(lr, model_name=f"linear_reg_item_{j+1}")

            pred = lr.predict(thetas.reshape(-1, 1))
            resid = y - pred
            g0, g1 = resid[group == 0], resid[group == 1]
            if len(g0) < 2 or len(g1) < 2:
                p, mean_diff = np.nan, np.nan
            else:
                _, p = ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
                mean_diff = np.mean(g1) - np.mean(g0)
            results.append({
                "Item": j + 1,
                "p_value": p,
                "mean_resid_diff": mean_diff
            })
        return pd.DataFrame(results)

    # ----------------------------------------------------------------------
    # REPORT
    # ----------------------------------------------------------------------
    def generate_data_profile(
        self,
        filename: str = "parameter_profile.md",
        include_plots: bool = True,
        mlflow_log: bool = True,
    ):
        """Generate markdown diagnostics report and log to MLflow."""
        parts = ["# Parameter Diagnostics Report\n"]
        parts.append("Generated by ParameterDiagnostics\n")

        order = self.check_order()
        ent = self.compute_entropy()
        info, _ = self.compute_item_information()

        parts.append(
            f"## Order Check\n- Disordered: {order['n_disordered']}\n")
        parts.append(
            f"## Entropy\n- Mean Entropy: {ent['overall_mean_entropy']:.4f}\n")

        discr = self.theta_discrimination()
        info_df = pd.DataFrame({
            "Item": np.arange(1, self.n_items + 1),
            "Mean_Info": info.mean(axis=1),
            "Theta_Discrimination": discr,
        })
        csv_path = os.path.join(self.output_dir, "item_info.csv")
        info_df.to_csv(csv_path, index=False)

        if include_plots:
            _, html_path = self.plot_cccs()
            parts.append(f"## CCCs\n- File: {os.path.basename(html_path)}\n")

        md_path = os.path.join(self.output_dir, filename)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))

        if mlflow_log:
            try:
                import mlflow
                mlflow.log_artifact(md_path)
                mlflow.log_artifact(csv_path)
                print("MLflow diagnostics report logged.")
            except Exception as e:
                print("MLflow artifact logging failed:", e)

        return md_path

    # ----------------------------------------------------------------------
    # DISCRIMINATION
    # ----------------------------------------------------------------------
    def theta_discrimination(self, thetas=None):
        """Numeric discrimination at θ=0."""
        P = self.compute_item_probabilities(thetas)
        k = np.arange(P.shape[2])
        E = (P * k[None, None, :]).sum(axis=2)
        dE = np.gradient(E, self.thetas_grid, axis=1)
        idx0 = np.argmin(np.abs(self.thetas_grid))
        return dE[:, idx0]
