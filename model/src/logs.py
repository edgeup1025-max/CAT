import psutil, os, time, threading, platform, socket, mlflow, gc
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import hashlib

__all__ = [
    "log_realtime_system_metrics", "monitor_stage", "memory_cleanup",
    "generate_plotly_performance_dashboard", "log_processed_dataset"
]


def log_realtime_system_metrics(run, interval=2):
    process = psutil.Process(os.getpid())

    def _loop():
        while getattr(run, "active", True):
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                mem_mb = process.memory_info().rss / 1024**2
                virt = psutil.virtual_memory()
                io = psutil.disk_io_counters()
                mlflow.log_metric("CPU_Usage_Percent", cpu_percent)
                mlflow.log_metric("Memory_MB", mem_mb)
                mlflow.log_metric("System_Memory_Used_MB", virt.used / 1024**2)
                mlflow.log_metric("Disk_Read_MB_Total",
                                  io.read_bytes / 1024**2)
                mlflow.log_metric("Disk_Write_MB_Total",
                                  io.write_bytes / 1024**2)
                time.sleep(interval)
            except Exception:
                break

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


# ----------------------------------------------------------------------
# 3️⃣  Stage-level profiling wrapper
# ----------------------------------------------------------------------
def monitor_stage(stage_name, func, *args, **kwargs):
    start_time = time.time()
    start_mem = get_memory_usage_mb()
    start_io = psutil.disk_io_counters()

    result = func(*args, **kwargs)

    end_time = time.time()
    end_mem = get_memory_usage_mb()
    end_io = psutil.disk_io_counters()

    dur = end_time - start_time
    mem_diff = end_mem - start_mem
    read_diff = (end_io.read_bytes - start_io.read_bytes) / 1024**2
    write_diff = (end_io.write_bytes - start_io.write_bytes) / 1024**2

    mlflow.log_param(f"{stage_name}_Duration_Sec", dur)
    mlflow.log_param(f"{stage_name}_Memory_Diff_MB", mem_diff)
    mlflow.log_param(f"{stage_name}_Disk_Read_MB", read_diff)
    mlflow.log_param(f"{stage_name}_Disk_Write_MB", write_diff)

    metrics = dict(
        Stage=stage_name,
        Duration_Sec=round(dur, 3),
        Memory_Diff_MB=round(mem_diff, 2),
        Disk_Read_MB=round(read_diff, 2),
        Disk_Write_MB=round(write_diff, 2),
    )
    return result, metrics


# ----------------------------------------------------------------------
# 4️⃣  Garbage-collector helper
# ----------------------------------------------------------------------
def memory_cleanup(label=""):
    proc = psutil.Process(os.getpid())
    before = proc.memory_info().rss / 1024**2
    gc.collect()
    after = proc.memory_info().rss / 1024**2
    print(f"[{label}] Memory: {before:.2f} → {after:.2f} MB")


# ----------------------------------------------------------------------
# 5️⃣  Plotly interactive artifact
# ----------------------------------------------------------------------
def generate_plotly_performance_dashboard(
        stage_metrics, output_html="system_performance_dashboard.html"):
    """
    Create an interactive Plotly dashboard combining CPU, Memory, Disk I/O across stages.
    Logs it as an MLflow artifact.
    Args:
        stage_metrics (list[dict]): metrics collected from monitor_stage
        output_html (str): file path to save HTML dashboard
    """
    df = pd.DataFrame(stage_metrics)
    stages = df["Stage"]

    fig = go.Figure()

    # Memory difference per stage
    fig.add_trace(
        go.Bar(x=stages,
               y=df["Memory_Diff_MB"],
               name="Memory Diff (MB)",
               marker_color="cornflowerblue"))

    # Disk I/O
    fig.add_trace(
        go.Bar(x=stages,
               y=df["Disk_Read_MB"],
               name="Disk Read (MB)",
               marker_color="lightgreen"))
    fig.add_trace(
        go.Bar(x=stages,
               y=df["Disk_Write_MB"],
               name="Disk Write (MB)",
               marker_color="orange"))

    # Duration line
    fig.add_trace(
        go.Scatter(x=stages,
                   y=df["Duration_Sec"],
                   name="Duration (sec)",
                   mode="lines+markers",
                   yaxis="y2",
                   line=dict(color="red", width=3)))

    # Layout styling
    fig.update_layout(title="System Performance Summary by Stage",
                      xaxis=dict(title="Pipeline Stages"),
                      yaxis=dict(title="Memory / Disk (MB)"),
                      yaxis2=dict(title="Duration (s)",
                                  overlaying="y",
                                  side="right"),
                      barmode="group",
                      legend=dict(x=0.01, y=1.15, orientation="h"),
                      template="plotly_dark",
                      height=500,
                      width=900)

    fig.write_html(output_html)
    mlflow.log_artifact(output_html)
    print(f" Plotly dashboard saved as MLflow artifact: {output_html}")


def log_processed_dataset(
        df: pd.DataFrame,
        name="processed_data",
        preview_rows=100,
        save_dir=r"D:\WORKSPACE\OFIICE_WORKS\model\src\saver"):
    try:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # ---- Compute dataset hash ----
        df_hash = hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values).hexdigest()

        mlflow.log_param(f"{name}_hash", df_hash)
        mlflow.log_param(f"{name}_num_rows", len(df))
        mlflow.log_param(f"{name}_num_columns", df.shape[1])

        # ---- Save full dataset (Parquet) ----
        parquet_path = save_path / f"{name}.parquet"
        df.to_parquet(parquet_path, index=False)
        mlflow.log_artifact(str(parquet_path))

        # ---- Save preview (first N rows) ----
        preview_path = save_path / f"{name}_preview.csv"
        df.head(preview_rows).to_csv(preview_path, index=False)
        mlflow.log_artifact(str(preview_path))

        print(f"[INFO] Logged dataset '{name}'")
        print(f"       → Full:   {parquet_path}")
        print(f"       → Preview:{preview_path}")
        print(f"       → Rows:   {len(df)}, Cols: {df.shape[1]}")
        print(f"       → Hash:   {df_hash[:12]}...")

    except Exception as e:
        print(f"[ERROR] Dataset logging failed for '{name}': {e}")
