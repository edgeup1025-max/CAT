import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from pathlib import Path
import warnings


def batch_spearman(data, batch_size=1000):
    """
    Compute Spearman correlation efficiently for large datasets.
    """
    from scipy.stats import rankdata
    if isinstance(data, pd.DataFrame):
        data = data.values
    ranked_data = np.zeros_like(data, dtype=float)
    for i in range(data.shape[1]):
        ranked_data[:, i] = rankdata(data[:, i])
    corr, _ = spearmanr(ranked_data)
    return corr


def run_feature_selection(thetas_df: pd.DataFrame,
                          saver_path: Path,
                          mlflow_active_run=None):
    """
    Run Spearman correlation and PCA feature analysis, log all results to MLflow.
    Clean, warning-free version with explicit model logging and reproducible environment.
    """
    # -----------------------------
    # Safety setup
    # -----------------------------
    warnings.filterwarnings("ignore",
                            message="Model was missing function: predict")

    saver_path.mkdir(parents=True, exist_ok=True)

    # Use the provided MLflow run context, or start a new one
    if mlflow_active_run is None:
        run_ctx = mlflow.start_run(run_name="Feature_Selection")
    else:
        run_ctx = mlflow_active_run

    with run_ctx:

        corr_matrix = batch_spearman(thetas_df)
        mean_corr = float(
            np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, 1)])))
        mlflow.log_metric("mean_spearman_corr", mean_corr)

        corr_df = pd.DataFrame(corr_matrix,
                               index=thetas_df.columns,
                               columns=thetas_df.columns)
        corr_path = saver_path / "spearman_corr.csv"
        corr_df.to_csv(corr_path)
        mlflow.log_artifact(str(corr_path))

        pca = PCA()
        pca.fit(thetas_df.values)

        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        n_90 = np.argmax(cumulative >= 0.9) + 1
        n_95 = np.argmax(cumulative >= 0.95) + 1

        mlflow.log_metric("pca_components_90pct", n_90)
        mlflow.log_metric("pca_components_95pct", n_95)

        # Save variance summary
        summary_df = pd.DataFrame({
            "Component": np.arange(1,
                                   len(explained) + 1),
            "Explained_Variance": explained,
            "Cumulative_Variance": cumulative
        })
        summary_path = saver_path / "pca_variance_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        mlflow.log_artifact(str(summary_path))

        # Log PCA model manually to avoid autolog warnings
        example_input = thetas_df.head(5)
        mlflow.sklearn.log_model(sk_model=pca,
                                 artifact_path="pca_model",
                                 input_example=example_input,
                                 pip_requirements=[
                                     "scikit-learn==1.7.2",
                                     "cloudpickle==3.1.2", "numpy>=1.26.0",
                                     "pandas>=2.2.0"
                                 ])

        pca_summary = {
            "n_components_90pct": n_90,
            "n_components_95pct": n_95,
            "explained_variance_ratio": explained.tolist(),
            "cumulative_variance": cumulative.tolist()
        }

        mlflow.log_dict(pca_summary, "pca_summary.json")

        print("Feature selection completed and logged successfully in MLflow")

        return pca_summary
