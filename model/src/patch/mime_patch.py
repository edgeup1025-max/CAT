import mimetypes

# Fix MIME type issues before MLflow starts
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.init()

import mlflow

if __name__ == "__main__":
    mlflow.ui.cli()
