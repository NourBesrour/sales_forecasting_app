# test_mlflow.py
import mlflow

mlflow.set_tracking_uri("file:./test_mlruns")

with mlflow.start_run():
    mlflow.log_param("test_param", 1)
    mlflow.log_metric("test_metric", 0.5)
    print("MLflow run logged successfully to ./test_mlruns")