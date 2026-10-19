"""Deploy a completed SageMaker model to a real-time endpoint."""
import argparse, os
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from config.project_config import SAGEMAKER_ROLE_ARN, SAGEMAKER_ENDPOINT_NAME

p=argparse.ArgumentParser()
p.add_argument("--model-artifact", required=True, help="s3://.../model.tar.gz")
p.add_argument("--instance-type", default="ml.m5.large")
p.add_argument("--endpoint-name", default=SAGEMAKER_ENDPOINT_NAME)
a=p.parse_args()

role=SAGEMAKER_ROLE_ARN or sagemaker.get_execution_role()
model=SKLearnModel(
    model_data=a.model_artifact,
    role=role,
    entry_point="inference.py",
    source_dir="sagemaker",
    framework_version="1.2-1",
    py_version="py3",
)
predictor=model.deploy(
    initial_instance_count=1,
    instance_type=a.instance_type,
    endpoint_name=a.endpoint_name,
)
print(predictor.endpoint_name)
