"""Invoke a SageMaker endpoint with a JSON feature object and print its response."""
import argparse, json
from pathlib import Path
import boto3
from config.project_config import AWS_REGION, SAGEMAKER_ENDPOINT_NAME

def main():
    """Parse command-line options and execute the script entry point."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint-name',default=SAGEMAKER_ENDPOINT_NAME)
    p.add_argument('--payload-json',required=True,help='Path to JSON containing {"features": {feature_name: numeric_value, ...}}')
    p.add_argument('--region',default=AWS_REGION)
    a=p.parse_args()
    payload=json.loads(Path(a.payload_json).read_text(encoding='utf-8'))
    response=boto3.client('sagemaker-runtime',region_name=a.region).invoke_endpoint(EndpointName=a.endpoint_name,ContentType='application/json',Accept='application/json',Body=json.dumps(payload).encode('utf-8'))
    print(response['Body'].read().decode('utf-8'))
if __name__=='__main__': main()
