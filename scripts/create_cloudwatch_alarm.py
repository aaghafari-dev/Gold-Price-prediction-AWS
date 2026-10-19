"""Create a basic SageMaker endpoint 5xx invocation alarm."""
import argparse
import boto3
from config.project_config import AWS_REGION, SAGEMAKER_ENDPOINT_NAME

def main():
    """Parse command-line options and execute the script entry point."""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint-name',default=SAGEMAKER_ENDPOINT_NAME)
    p.add_argument('--region',default=AWS_REGION)
    p.add_argument('--sns-topic-arn',default='',help='Optional existing SNS topic ARN for alarm actions')
    a=p.parse_args()
    cw=boto3.client('cloudwatch',region_name=a.region)
    args={'AlarmName':f'{a.endpoint_name}-Invocation5XXErrors','AlarmDescription':'Alert when the SageMaker endpoint emits 5xx invocation errors.','Namespace':'AWS/SageMaker','MetricName':'Invocation5XXErrors','Dimensions':[{'Name':'EndpointName','Value':a.endpoint_name},{'Name':'VariantName','Value':'AllTraffic'}],'Statistic':'Sum','Period':60,'EvaluationPeriods':1,'DatapointsToAlarm':1,'Threshold':1,'ComparisonOperator':'GreaterThanOrEqualToThreshold','TreatMissingData':'notBreaching'}
    if a.sns_topic_arn: args['AlarmActions']=[a.sns_topic_arn]
    cw.put_metric_alarm(**args)
    print(f"Created CloudWatch alarm {args['AlarmName']}")
if __name__=='__main__': main()
