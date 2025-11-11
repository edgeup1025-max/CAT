import json
from dotenv import load_dotenv
import re
import duckdb
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

__all__ = ["configure_duckdb_cloud_access", 'settings']


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env",
                                      env_file_encoding="utf-8")

    AWS_ACCESS_KEY_ID: str | None = None
    AWS_SECRET_ACCESS_KEY: str | None = None
    AWS_REGION: str = "us-east-1"

    GCS_ACCESS_TOKEN: str | None = None

    AZURE_STORAGE_ACCOUNT: str | None = None
    AZURE_STORAGE_ACCESS_KEY: str | None = None

    HTTP_BEARER_TOKEN: str | None = None


settings = Settings()


def configure_duckdb_cloud_access(con: duckdb.DuckDBPyConnection,
                                  file_path: str):
    """Automatically configure DuckDB based on file source (local/S3/GCS/Azure/HTTP)."""

    # Ensure HTTPFS extension is available
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")

    # Detect the source type
    if file_path.startswith("s3://"):
        print(" Configuring AWS S3 access...")
        con.execute(
            f"SET s3_access_key_id='{os.getenv('AWS_ACCESS_KEY_ID', '')}';")
        con.execute(
            f"SET s3_secret_access_key='{os.getenv('AWS_SECRET_ACCESS_KEY', '')}';"
        )
        con.execute(f"SET s3_region='{os.getenv('AWS_REGION', 'us-east-1')}';")

    elif file_path.startswith("gs://"):
        print("Configuring Google Cloud Storage access...")
        token = os.getenv("GCS_ACCESS_TOKEN")
        if token:
            con.execute(f"SET gcs_access_token='{token}';")

    elif file_path.startswith("azure://"):
        print(" Configuring Azure Blob Storage access...")
        con.execute(
            f"SET azure_storage_account='{os.getenv('AZURE_STORAGE_ACCOUNT', '')}';"
        )
        con.execute(
            f"SET azure_storage_access_key='{os.getenv('AZURE_STORAGE_ACCESS_KEY', '')}';"
        )

    elif re.match(r"^https?://", file_path):
        print(" Configuring HTTP(S) access...")
        # Optional: set headers for auth APIs
        token = os.getenv("HTTP_BEARER_TOKEN")
        if token:
            con.execute(f"SET http_headers='Authorization: Bearer {token}';")

    else:
        print(" Local file detected — no cloud setup needed.")

    print(" DuckDB configuration completed.")
