"""Upsert/start SageMaker Pipeline and associate the run with SageMaker Experiments."""
import importlib.util
from datetime import datetime, timezone
import boto3
from config.project_config import AWS_REGION, SAGEMAKER_MODEL_PACKAGE_GROUP


def main():
    """Parse command-line options and execute the script entry point."""
    spec = importlib.util.spec_from_file_location("gold_price_sagemaker_pipeline", "sagemaker/pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sm = boto3.client("sagemaker", region_name=AWS_REGION)
    try:
        sm.describe_model_package_group(ModelPackageGroupName=SAGEMAKER_MODEL_PACKAGE_GROUP)
    except sm.exceptions.ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code not in {"ResourceNotFound", "ValidationException"}:
            raise
        sm.create_model_package_group(ModelPackageGroupName=SAGEMAKER_MODEL_PACKAGE_GROUP, ModelPackageGroupDescription="Ironhack Gold Price Prediction V7 educational candidates")

    experiment_name = "GoldPricePredictionV7"
    try:
        sm.describe_experiment(ExperimentName=experiment_name)
    except sm.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in {"ResourceNotFound", "ValidationException"}:
            raise
        sm.create_experiment(ExperimentName=experiment_name, DisplayName="Gold Price Prediction V7", Description="Ironhack end-to-end gold-futures proxy forecasting project")

    module.pipeline.upsert(role_arn=module.ROLE)
    trial_name = "gold-v7-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    execution = module.pipeline.start(experiment_config={"ExperimentName": experiment_name, "TrialName": trial_name})
    print(f"Pipeline={module.pipeline.name}\nExecutionArn={execution.arn}\nExperiment={experiment_name}\nTrial={trial_name}")

if __name__ == "__main__": main()
