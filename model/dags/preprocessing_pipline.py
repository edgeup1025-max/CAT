from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from src.load_data import local_duckdb, config_reader
from src.dependency import folder_lookup


def load_data(config_path: str, **context):
    """
    Task: Load data into DuckDB based on a given config file.
    Context (execution_date, run_id, etc.) is automatically available.
    """
    # Step 1: Resolve and validate config path
    resolved_path = folder_lookup(config_path)
    print(f"Resolved config path: {resolved_path}")

    # Step 2: Read configuration file
    args = config_reader(resolved_path)
    print(f"Loaded configuration: {args}")

    # Step 3: Load data into DuckDB
    data = local_duckdb(args["FILE_PATH"], args["LIMIT"])
    print(f"✅ Loaded {len(data)} records into DuckDB successfully.")

    # Optional: Push results to XCom
    context["ti"].xcom_push(key="record_count", value=len(data))


# Define the DAG
with DAG(
        dag_id="user_typing_input",
        start_date=datetime(2024, 1, 1),
        schedule=None,
        catchup=False,
        params={
            "config":
            r"D:\WORKSPACE\OFIICE_WORKS\model\src\config_data\data_config.cfg"
        },
        description="DAG to load data into DuckDB based on user config",
        tags=["duckdb", "data_load"],
) as dag:

    load_data_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data,
        op_args=["{{ params.config }}"],  # Pass Jinja parameter
    )

    load_data_task
