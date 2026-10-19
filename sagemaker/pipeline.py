"""project-sized SageMaker Pipeline: Processing -> Training -> Evaluation -> quality gate -> Registry."""
import boto3
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import TrainingStep, ProcessingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo, ConditionEquals
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from config.project_config import AWS_REGION, SAGEMAKER_ROLE_ARN, SAGEMAKER_MODEL_PACKAGE_GROUP, S3_BUCKET, S3_PREFIX

ROLE = SAGEMAKER_ROLE_ARN or sagemaker.get_execution_role()
BUCKET_PREFIX = f"s3://{S3_BUCKET}/{S3_PREFIX}" if S3_BUCKET else "s3://YOUR_BUCKET/gold-price-prediction"
RAW_DATA = ParameterString(name="RawMarketDataS3Uri", default_value=f"{BUCKET_PREFIX}/raw/market/bootstrap_market_history.csv")
MODEL_NAME = ParameterString(name="ModelName", default_value="Random Forest")
sm_session = sagemaker.Session(boto_session=boto3.Session(region_name=AWS_REGION))

# Processing performs schema checks, feature engineering, and a chronological 80/20 split.
processor = SKLearnProcessor(
    framework_version="1.2-1", role=ROLE, instance_type="ml.m5.large", instance_count=1,
    base_job_name="gold-price-preprocess", sagemaker_session=sm_session,
)
process_args = processor.run(
    code="preprocess.py", source_dir="sagemaker",
    inputs=[ProcessingInput(input_name="raw", source=RAW_DATA, destination="/opt/ml/processing/input")],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
    ],
)
preprocess_step = ProcessingStep(name="PrepareGoldFeatures", step_args=process_args)

estimator = SKLearn(
    entry_point="train.py", source_dir="sagemaker", role=ROLE,
    instance_type="ml.m5.large", instance_count=1, framework_version="1.2-1", py_version="py3",
    output_path=f"{BUCKET_PREFIX}/models", hyperparameters={"model-name": MODEL_NAME},
    sagemaker_session=sm_session,
)
training_step = TrainingStep(
    name="TrainGoldModel", estimator=estimator,
    inputs={"train": TrainingInput(
        s3_data=preprocess_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
        content_type="text/csv")},
)

evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
evaluation_args = processor.run(
    code="evaluate.py", source_dir="sagemaker",
    inputs=[
        ProcessingInput(input_name="test", source=preprocess_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test"),
        ProcessingInput(input_name="model", source=training_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
    ],
    outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
)
evaluation_step = ProcessingStep(name="EvaluateGoldModel", step_args=evaluation_args, property_files=[evaluation_report])

inference_model = estimator.create_model(entry_point="inference.py", source_dir="sagemaker")
register_step = RegisterModel(
    name="RegisterGoldModel", model=inference_model,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["application/json"], response_types=["application/json"],
    inference_instances=["ml.m5.large"], transform_instances=["ml.m5.large"],
    model_package_group_name=SAGEMAKER_MODEL_PACKAGE_GROUP,
    approval_status="PendingManualApproval",
)
quality_gate = ConditionStep(
    name="CheckGoldModelQuality",
    conditions=[
        ConditionEquals(left=JsonGet(step_name=evaluation_step.name, property_file=evaluation_report, json_path="beats_baseline"), right=True),
        ConditionGreaterThanOrEqualTo(left=JsonGet(step_name=evaluation_step.name, property_file=evaluation_report, json_path="improvement_pct"), right=0.0),
    ],
    if_steps=[register_step], else_steps=[],
)

pipeline = Pipeline(
    name="GoldPricePredictionPipeline", parameters=[RAW_DATA, MODEL_NAME],
    steps=[preprocess_step, training_step, evaluation_step, quality_gate], sagemaker_session=sm_session,
)

if __name__ == "__main__":
    print(pipeline.definition())
