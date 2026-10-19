"""Deploy only an Approved SageMaker Model Registry package to a real-time endpoint."""
import argparse
import boto3
from config.project_config import AWS_REGION, SAGEMAKER_ROLE_ARN, SAGEMAKER_ENDPOINT_NAME

def main():
    """Parse command-line options and execute the script entry point."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-package-arn', required=True)
    p.add_argument('--role-arn', default=SAGEMAKER_ROLE_ARN)
    p.add_argument('--endpoint-name', default=SAGEMAKER_ENDPOINT_NAME)
    p.add_argument('--instance-type', default='ml.m5.large')
    p.add_argument('--region', default=AWS_REGION)
    a=p.parse_args()
    if not a.role_arn: p.error('Supply --role-arn or set SAGEMAKER_ROLE_ARN.')
    sm=boto3.client('sagemaker',region_name=a.region)
    pkg=sm.describe_model_package(ModelPackageArn=a.model_package_arn)
    if pkg.get('ModelApprovalStatus') != 'Approved':
        raise SystemExit(f"Refusing deployment: model package status is {pkg.get('ModelApprovalStatus')!r}, not 'Approved'.")
    containers=pkg.get('InferenceSpecification',{}).get('Containers',[])
    if not containers: raise RuntimeError('Approved package has no inference container specification.')
    model_name=(a.endpoint_name+'-model')[:63]
    config_name=(a.endpoint_name+'-config')[:63]
    container={k:v for k,v in containers[0].items() if k in {'Image','ModelDataUrl','Environment','ImageConfig','AdditionalModelDataSources'}}
    sm.create_model(ModelName=model_name,ExecutionRoleArn=a.role_arn,PrimaryContainer=container)
    sm.create_endpoint_config(EndpointConfigName=config_name,ProductionVariants=[{'VariantName':'AllTraffic','ModelName':model_name,'InitialInstanceCount':1,'InstanceType':a.instance_type,'InitialVariantWeight':1.0}])
    sm.create_endpoint(EndpointName=a.endpoint_name,EndpointConfigName=config_name)
    waiter=sm.get_waiter('endpoint_in_service'); waiter.wait(EndpointName=a.endpoint_name)
    print(f"Endpoint is InService: {a.endpoint_name}")
if __name__=='__main__': main()
