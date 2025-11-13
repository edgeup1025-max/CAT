import logging
from logs import (
    log_realtime_system_metrics,
    monitor_stage,
    log_processed_dataset,
    generate_plotly_performance_dashboard,
)
from load_data import local_duckdb, irt_data_processor
from preprocessing import Preprocessing
from dependency import memory_cleanup
from irt.PolychotomousIRT import GRM
from Diagnostics import ParameterDiagnostics
import mlflow
import mlflow.data as md
from pathlib import Path
import pandas as pd
import warnings
import numpy as np
from dotenv import load_dotenv


# ======================
# Logger Setup
# ======================
def setup_logger(log_dir: Path):
    """Configure logger to output to console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "pipeline.log"

    logger = logging.getLogger("MLPipeline")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", "%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    load_dotenv()
    warnings.filterwarnings("ignore")

    FILE_PATH = r"C:\Users\rohith\Downloads\archive\IPIP-FFM-data-8Nov2018\data-final.csv"
    LIMIT = 100000
    SAVER_PATH = Path(r"D:\WORKSPACE\OFIICE_WORKS\model\src\saver")
    TRACKING_URI = Path(r"D:\WORKSPACE\OFIICE_WORKS\model\src\MLflow")

    logger = setup_logger(SAVER_PATH)
    logger.info("Starting GRM + Feature Selection pipeline...")

    # ======================
    # MLflow Tracking Setup
    # ======================
    mlflow.set_tracking_uri(f"file:///{TRACKING_URI.as_posix()}")
    mlflow.set_experiment("Dataset_Versioning_And_Monitoring")

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(
            run_name="Preprocessing_GRM_Feature_Selection") as run:
        logger.info(f"MLflow run started: {run.info.run_id}")
        monitor_thread = log_realtime_system_metrics(run, interval=1)

        # ======================
        # Stage 1: Load Dataset
        # ======================
        logger.info("Loading dataset...")
        df, m1 = monitor_stage("DuckDB_Load",
                               local_duckdb,
                               FILE_PATH,
                               limit=LIMIT)
        logger.info(f" Dataset loaded successfully: {df.shape}")

        # ======================
        # Stage 2: Preprocessing
        # ======================
        logger.info("🧹 Running preprocessing pipeline...")
        pre = Preprocessing(use_ordinal_for_categorical=True)
        (processed_df, likert_df), m2 = monitor_stage("Preprocessing",
                                                      pre.fit_transform, df)
        logger.info(f"Preprocessing complete. Shape: {processed_df.shape}")

        if processed_df is not None:
            log_processed_dataset(processed_df, name="processed_data")
        if likert_df is not None:
            log_processed_dataset(likert_df, name="likert_data")

        # ======================
        # Stage 3: GRM Training
        # ======================
        logger.info("Starting GRM model training for Big Five traits...")

        O_items = likert_df.iloc[:, 40:50].values.astype(int)
        C_items = likert_df.iloc[:, 30:40].values.astype(int)
        E_items = likert_df.iloc[:, 0:10].values.astype(int)
        A_items = likert_df.iloc[:, 10:20].values.astype(int)
        N_items = likert_df.iloc[:, 20:30].values.astype(int)

        traits = {
            "Openness": O_items,
            "Conscientiousness": C_items,
            "Extraversion": E_items,
            "Agreeableness": A_items,
            "Neuroticism": N_items,
        }

        models, metrics_list, theta_dict = {}, [], {}

        for trait, X_trait in traits.items():
            logger.info(f"Training GRM for {trait}...")
            with mlflow.start_run(run_name=f"{trait}_GRM_FIT",
                                  nested=True) as trait_run:
                log_data = md.from_pandas(
                    pd.DataFrame(
                        X_trait,
                        columns=[
                            f"{trait}_{i+1}" for i in range(X_trait.shape[1])
                        ],
                    ).astype("float64"),
                    source="local",
                    name=f"{trait}_training_data",
                )
                mlflow.log_input(log_data, context="training")

                model = GRM(
                    n_items=X_trait.shape[1],
                    n_categories=5,
                    learning_rate_alpha_beta=1e-3,
                    learning_rate_theta=5e-3,
                    n_epochs=100,
                    l2_lambda=1e-4,
                    patience=30,
                    track_with_mlflow=True,
                )

                model, m = monitor_stage(f"{trait}_GRM_FIT", model.fit,
                                         X_trait)
                models[trait] = model
                metrics_list.append(m)
                logger.info(f" GRM training complete for {trait}")

                # Extract IRT parameters
                thetas, Alpha, betas = irt_data_processor(
                    model.thetas_, model.alpha, model.beta)
                logger.info(f"Saving IRT parameters for {trait}...")

                # Save artifacts
                thetas.to_csv(SAVER_PATH / f"{trait}_thetas.csv", index=False)
                Alpha.to_csv(SAVER_PATH / f"{trait}_Alpha.csv", index=False)
                betas.to_csv(SAVER_PATH / f"{trait}_Betas.csv", index=False)

                mlflow.log_artifact(SAVER_PATH / f"{trait}_thetas.csv")
                mlflow.log_artifact(SAVER_PATH / f"{trait}_Alpha.csv")
                mlflow.log_artifact(SAVER_PATH / f"{trait}_Betas.csv")

                theta_dict[trait] = model.thetas_.flatten()

        # Combine theta data across traits
        thetas_df = pd.DataFrame(theta_dict)
        combined_path = SAVER_PATH / "thetas_combined.csv"
        thetas_df.to_csv(combined_path, index=False)
        mlflow.log_artifact(combined_path)
        logger.info("Combined theta dataframe saved and logged.")

        # ======================
        # Stage 4: Diagnostics
        # ======================
        logger.info("Running parameter diagnostics...")
        diag = ParameterDiagnostics(beta=betas, alpha=Alpha)
        md_report = diag.generate_data_profile(filename="parameter_profile.md",
                                               include_plots=True,
                                               mlflow_log=True)
        mlflow.log_artifact(md_report)
        logger.info("Diagnostics report logged successfully.")

        # ======================
        # Stage 6: Dashboard + Cleanup
        # ======================
        logger.info("Generating performance dashboard...")
        generate_plotly_performance_dashboard(
            [m1, m2, *metrics_list],
            output_html=SAVER_PATH / "performance_dashboard.html",
        )
        mlflow.log_artifact(SAVER_PATH / "performance_dashboard.html")
        logger.info("Performance dashboard logged successfully.")

        # Cleanup
        memory_cleanup("After feature selection")
        monitor_thread.join(timeout=2)

        logger.info("Pipeline execution completed successfully.")
