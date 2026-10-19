"""Register a SageMaker model package. Use after a validated training run."""
import argparse, boto3, os
from config.project_config import AWS_REGION, SAGEMAKER_MODEL_PACKAGE_GROUP

p=argparse.ArgumentParser()
p.add_argument("--model-data", required=True)
p.add_argument("--image-uri", required=True)
p.add_argument("--model-package-group", default=SAGEMAKER_MODEL_PACKAGE_GROUP)
p.add_argument("--approval-status", default="PendingManualApproval",
               choices=["PendingManualApproval","Approved","Rejected"])
a=p.parse_args()

sm=boto3.client("sagemaker", region_name=AWS_REGION)
resp=sm.create_model_package(
    ModelPackageGroupName=a.model_package_group,
    ModelApprovalStatus=a.approval_status,
    InferenceSpecification={
        "Containers":[{"Image":a.image_uri,"ModelDataUrl":a.model_data}],
        "SupportedContentTypes":["application/json"],
        "SupportedResponseMIMETypes":["application/json"],
    },
)
print(resp["ModelPackageArn"])
