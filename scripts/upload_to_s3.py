"""Upload selected project data and artifacts to the configured S3 bucket for the optional SageMaker workflow."""
import sys
from pathlib import Path as _ProjectPath
sys.path.insert(0, str(_ProjectPath(__file__).resolve().parents[1]))
import argparse, boto3
from config.project_config import AWS_REGION

p=argparse.ArgumentParser()
p.add_argument("--file", required=True)
p.add_argument("--bucket", required=True)
p.add_argument("--key", required=True)
a=p.parse_args()
boto3.client("s3", region_name=AWS_REGION).upload_file(a.file,a.bucket,a.key)
print(f"s3://{a.bucket}/{a.key}")
