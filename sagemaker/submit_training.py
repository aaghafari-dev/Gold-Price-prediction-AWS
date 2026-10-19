"""Submit a managed SageMaker training job using the scikit-learn container."""
import os, json, argparse
import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from config.project_config import AWS_REGION, SAGEMAKER_ROLE_ARN, S3_BUCKET, S3_PREFIX

p=argparse.ArgumentParser()
p.add_argument("--train-s3", required=True)
p.add_argument("--model-name", default="XGBoost")
p.add_argument("--instance-type", default="ml.m5.large")
p.add_argument("--wait", action="store_true")
a=p.parse_args()

session=sagemaker.Session()
role=SAGEMAKER_ROLE_ARN or sagemaker.get_execution_role()
estimator=SKLearn(
    entry_point="train.py",
    source_dir="sagemaker",
    role=role,
    instance_type=a.instance_type,
    instance_count=1,
    framework_version="1.2-1",
    py_version="py3",
    hyperparameters={"model-name":a.model_name},
    output_path=f"s3://{S3_BUCKET}/{S3_PREFIX}/models",
    sagemaker_session=session,
)
estimator.fit({"train": a.train_s3}, wait=a.wait)
print(estimator.latest_training_job.name)
