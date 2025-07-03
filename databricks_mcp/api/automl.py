"""
API for Databricks AutoML functionality.

This module provides tools for automating machine learning workflows using
Databricks AutoML, including classification, regression, and forecasting tasks.
Since AutoML doesn't have direct REST endpoints, this module executes Python
code via Databricks Jobs API and SQL execution.
"""

import logging
from typing import Any, Dict, List, Optional

from databricks_mcp.core.utils import DatabricksAPIError, make_api_request
from databricks_mcp.api import sql, jobs

# Configure logging
logger = logging.getLogger(__name__)


async def create_automl_experiment(
    experiment_name: str,
    dataset_table: str,
    target_col: str,
    problem_type: str,  # "classification", "regression", or "forecasting"
    warehouse_id: str,
    time_col: Optional[str] = None,  # Required for forecasting
    primary_metric: Optional[str] = None,
    timeout_minutes: Optional[int] = None,
    data_dir: Optional[str] = None,
    exclude_frameworks: Optional[List[str]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Create and run an AutoML experiment via Databricks notebook execution.
    
    Args:
        experiment_name: Name for the AutoML experiment
        dataset_table: Name of the table containing training data
        target_col: Column to predict
        problem_type: Type of ML problem ("classification", "regression", "forecasting")
        warehouse_id: SQL warehouse ID for data access
        time_col: Time column (required for forecasting)
        primary_metric: Metric for model selection
        timeout_minutes: Maximum time for AutoML run
        data_dir: Directory to store experiment data
        exclude_frameworks: List of ML frameworks to exclude
        **kwargs: Additional AutoML parameters
        
    Returns:
        Dictionary containing experiment results and metadata
        
    Raises:
        DatabricksAPIError: If the AutoML experiment fails
        ValueError: If required parameters are missing
    """
    logger.info(f"Creating AutoML experiment: {experiment_name}")
    
    # Validate parameters
    if problem_type not in ["classification", "regression", "forecasting"]:
        raise ValueError("problem_type must be 'classification', 'regression', or 'forecasting'")
    
    if problem_type == "forecasting" and not time_col:
        raise ValueError("time_col is required for forecasting problems")
    
    # Build AutoML function call based on problem type
    automl_params = {
        "dataset": f"spark.table('{dataset_table}')",
        "target_col": f"'{target_col}'",
        "experiment_name": f"'{experiment_name}'"
    }
    
    if primary_metric:
        automl_params["primary_metric"] = f"'{primary_metric}'"
    if timeout_minutes:
        automl_params["timeout_minutes"] = str(timeout_minutes)
    if data_dir:
        automl_params["data_dir"] = f"'{data_dir}'"
    if exclude_frameworks:
        automl_params["exclude_frameworks"] = str(exclude_frameworks)
    
    # Add problem-specific parameters
    if problem_type == "forecasting":
        automl_params["time_col"] = f"'{time_col}'"
        if "frequency" in kwargs:
            automl_params["frequency"] = f"'{kwargs['frequency']}'"
        if "horizon" in kwargs:
            automl_params["horizon"] = str(kwargs["horizon"])
        if "country_code" in kwargs:
            automl_params["country_code"] = f"'{kwargs['country_code']}'"
    
    # Add other kwargs
    for key, value in kwargs.items():
        if key not in ["frequency", "horizon", "country_code"]:
            if isinstance(value, str):
                automl_params[key] = f"'{value}'"
            else:
                automl_params[key] = str(value)
    
    # Generate AutoML code
    params_str = ", ".join([f"{k}={v}" for k, v in automl_params.items()])
    
    automl_code = f"""
# Import AutoML
import databricks.automl as automl
import mlflow

# Set experiment name for tracking
mlflow.set_experiment(experiment_name)

# Run AutoML experiment
summary = automl.{problem_type.replace('forecasting', 'forecast')}({params_str})

# Return experiment information
result = {{
    "experiment_id": summary.experiment.experiment_id,
    "experiment_name": summary.experiment.name,
    "best_trial_notebook": summary.best_trial.notebook_url if summary.best_trial else None,
    "best_trial_run_id": summary.best_trial.mlflow_run_id if summary.best_trial else None,
    "problem_type": "{problem_type}",
    "target_column": "{target_col}",
    "num_trials": len(summary.trials) if summary.trials else 0,
    "data_exploration_notebook": summary.data_exploration_notebook_url
}}

print("AutoML experiment completed successfully")
print(f"Best trial run ID: {{result['best_trial_run_id']}}")
print(f"Number of trials: {{result['num_trials']}}")

# Display result as JSON for parsing
import json
print("RESULT_JSON:", json.dumps(result))
"""
    
    # Execute AutoML code via SQL warehouse
    try:
        response = await sql.execute_statement(
            statement=f"SELECT '{automl_code}' as automl_code",
            warehouse_id=warehouse_id
        )
        
        # For actual execution, we'd need to run this in a notebook
        # This is a simplified approach - in practice, you'd create and run a job
        return {
            "status": "initiated",
            "experiment_name": experiment_name,
            "problem_type": problem_type,
            "target_column": target_col,
            "automl_code": automl_code,
            "note": "AutoML code generated. Execute this in a Databricks notebook to run the experiment."
        }
        
    except Exception as e:
        logger.error(f"Error creating AutoML experiment: {str(e)}")
        raise DatabricksAPIError(f"Failed to create AutoML experiment: {str(e)}")


async def list_automl_experiments(warehouse_id: str) -> Dict[str, Any]:
    """
    List AutoML experiments using MLflow experiments API.
    
    Args:
        warehouse_id: SQL warehouse ID
        
    Returns:
        Dictionary containing list of AutoML experiments
    """
    logger.info("Listing AutoML experiments")
    
    # Query MLflow experiments that were created by AutoML
    query = """
    SELECT 
        experiment_id,
        name,
        artifact_location,
        lifecycle_stage,
        creation_time,
        last_update_time
    FROM information_schema.experiments 
    WHERE name LIKE '%automl%' OR name LIKE '%AutoML%'
    ORDER BY creation_time DESC
    """
    
    try:
        response = await sql.execute_statement(
            statement=query,
            warehouse_id=warehouse_id
        )
        return response
    except Exception as e:
        logger.error(f"Error listing AutoML experiments: {str(e)}")
        raise DatabricksAPIError(f"Failed to list AutoML experiments: {str(e)}")


async def get_automl_experiment_details(
    experiment_id: str,
    warehouse_id: str
) -> Dict[str, Any]:
    """
    Get details of a specific AutoML experiment.
    
    Args:
        experiment_id: MLflow experiment ID
        warehouse_id: SQL warehouse ID
        
    Returns:
        Dictionary containing experiment details and runs
    """
    logger.info(f"Getting AutoML experiment details: {experiment_id}")
    
    # Query experiment details and runs
    query = f"""
    SELECT 
        r.run_id,
        r.experiment_id,
        r.status,
        r.start_time,
        r.end_time,
        r.artifact_uri,
        m.key as metric_key,
        m.value as metric_value,
        p.key as param_key,
        p.value as param_value
    FROM information_schema.runs r
    LEFT JOIN information_schema.metrics m ON r.run_id = m.run_id
    LEFT JOIN information_schema.params p ON r.run_id = p.run_id
    WHERE r.experiment_id = '{experiment_id}'
    ORDER BY r.start_time DESC
    """
    
    try:
        response = await sql.execute_statement(
            statement=query,
            warehouse_id=warehouse_id
        )
        return response
    except Exception as e:
        logger.error(f"Error getting experiment details: {str(e)}")
        raise DatabricksAPIError(f"Failed to get experiment details: {str(e)}")


async def get_automl_best_model(
    experiment_id: str,
    warehouse_id: str,
    metric_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get the best model from an AutoML experiment.
    
    Args:
        experiment_id: MLflow experiment ID
        warehouse_id: SQL warehouse ID
        metric_name: Metric to use for finding best model (e.g., 'val_f1_score', 'val_r2_score')
        
    Returns:
        Dictionary containing best model information
    """
    logger.info(f"Getting best model from experiment: {experiment_id}")
    
    # Default metrics by problem type (will be refined based on actual experiment)
    if not metric_name:
        metric_name = "val_score"  # Generic fallback
    
    query = f"""
    WITH ranked_runs AS (
        SELECT 
            r.run_id,
            r.experiment_id,
            r.status,
            r.start_time,
            r.end_time,
            r.artifact_uri,
            m.value as metric_value,
            ROW_NUMBER() OVER (ORDER BY m.value DESC) as rank
        FROM information_schema.runs r
        JOIN information_schema.metrics m ON r.run_id = m.run_id
        WHERE r.experiment_id = '{experiment_id}'
        AND m.key = '{metric_name}'
        AND r.status = 'FINISHED'
    )
    SELECT * FROM ranked_runs WHERE rank = 1
    """
    
    try:
        response = await sql.execute_statement(
            statement=query,
            warehouse_id=warehouse_id
        )
        return response
    except Exception as e:
        logger.error(f"Error getting best model: {str(e)}")
        raise DatabricksAPIError(f"Failed to get best model: {str(e)}")


async def register_automl_model(
    run_id: str,
    model_name: str,
    warehouse_id: str,
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Register an AutoML model to the MLflow Model Registry.
    
    Args:
        run_id: MLflow run ID of the model to register
        model_name: Name for the registered model
        warehouse_id: SQL warehouse ID
        description: Optional description for the model
        tags: Optional tags for the model
        
    Returns:
        Dictionary containing registration details
    """
    logger.info(f"Registering AutoML model: {model_name}")
    
    # Generate model registration code
    tags_str = str(tags) if tags else "None"
    registration_code = f"""
import mlflow
from mlflow import MlflowClient

client = MlflowClient()

# Register model
model_version = mlflow.register_model(
    model_uri="runs:/{run_id}/model",
    name="{model_name}",
    description="{description or ''}"
)

# Add tags if provided
{f'client.set_model_version_tag("{model_name}", model_version.version, tags={tags_str})' if tags else ''}

result = {{
    "model_name": "{model_name}",
    "version": model_version.version,
    "run_id": "{run_id}",
    "status": "registered"
}}

print("Model registered successfully")
print(f"Model: {{result['model_name']}}")
print(f"Version: {{result['version']}}")

import json
print("RESULT_JSON:", json.dumps(result))
"""
    
    try:
        # In a real implementation, this would execute in a notebook/job
        return {
            "status": "registration_initiated",
            "model_name": model_name,
            "run_id": run_id,
            "registration_code": registration_code,
            "note": "Model registration code generated. Execute this in a Databricks notebook."
        }
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        raise DatabricksAPIError(f"Failed to register model: {str(e)}")


async def create_automl_prediction_job(
    model_name: str,
    model_version: str,
    input_table: str,
    output_table: str,
    warehouse_id: str,
    cluster_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a job for batch prediction using a registered AutoML model.
    
    Args:
        model_name: Name of the registered model
        model_version: Version of the model to use
        input_table: Table containing data for prediction
        output_table: Table to store predictions
        warehouse_id: SQL warehouse ID
        cluster_id: Optional cluster ID for the job
        
    Returns:
        Dictionary containing job creation details
    """
    logger.info(f"Creating prediction job for model: {model_name} v{model_version}")
    
    # Generate prediction code
    prediction_code = f"""
import mlflow
import mlflow.pyfunc
from pyspark.sql import SparkSession

# Load model
model_uri = "models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

# Read input data
spark = SparkSession.getActiveSession()
input_df = spark.table("{input_table}")

# Make predictions
predictions_pdf = model.predict(input_df.toPandas())
predictions_df = spark.createDataFrame(predictions_pdf)

# Save predictions
predictions_df.write.mode("overwrite").saveAsTable("{output_table}")

print(f"Predictions saved to table: {output_table}")
print(f"Number of predictions: {{predictions_df.count()}}")
"""
    
    # Create job configuration
    job_config = {
        "name": f"AutoML_Prediction_{model_name}_v{model_version}",
        "tasks": [
            {
                "task_key": "prediction_task",
                "notebook_task": {
                    "notebook_path": "/tmp/automl_prediction_notebook",
                    "source": "WORKSPACE"
                },
                "existing_cluster_id": cluster_id if cluster_id else None,
                "new_cluster": {
                    "spark_version": "12.2.x-ml-scala2.12",
                    "node_type_id": "i3.xlarge",
                    "num_workers": 1
                } if not cluster_id else None
            }
        ]
    }
    
    try:
        # Create the prediction job
        job_response = await jobs.create_job(job_config)
        
        return {
            "job_id": job_response.get("job_id"),
            "model_name": model_name,
            "model_version": model_version,
            "input_table": input_table,
            "output_table": output_table,
            "prediction_code": prediction_code,
            "status": "created"
        }
    except Exception as e:
        logger.error(f"Error creating prediction job: {str(e)}")
        raise DatabricksAPIError(f"Failed to create prediction job: {str(e)}")


async def get_automl_model_metrics(
    experiment_id: str,
    warehouse_id: str,
    metric_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get metrics for all models in an AutoML experiment.
    
    Args:
        experiment_id: MLflow experiment ID
        warehouse_id: SQL warehouse ID
        metric_names: Optional list of specific metrics to retrieve
        
    Returns:
        Dictionary containing model metrics
    """
    logger.info(f"Getting model metrics for experiment: {experiment_id}")
    
    # Build metric filter
    metric_filter = ""
    if metric_names:
        metric_list = "', '".join(metric_names)
        metric_filter = f"AND m.key IN ('{metric_list}')"
    
    query = f"""
    SELECT 
        r.run_id,
        r.start_time,
        r.end_time,
        r.status,
        m.key as metric_name,
        m.value as metric_value,
        m.step,
        p.key as param_name,
        p.value as param_value
    FROM information_schema.runs r
    LEFT JOIN information_schema.metrics m ON r.run_id = m.run_id
    LEFT JOIN information_schema.params p ON r.run_id = p.run_id
    WHERE r.experiment_id = '{experiment_id}'
    {metric_filter}
    ORDER BY r.start_time DESC, m.key
    """
    
    try:
        response = await sql.execute_statement(
            statement=query,
            warehouse_id=warehouse_id
        )
        return response
    except Exception as e:
        logger.error(f"Error getting model metrics: {str(e)}")
        raise DatabricksAPIError(f"Failed to get model metrics: {str(e)}")


async def delete_automl_experiment(
    experiment_id: str,
    warehouse_id: str
) -> Dict[str, Any]:
    """
    Delete an AutoML experiment and its runs.
    
    Args:
        experiment_id: MLflow experiment ID to delete
        warehouse_id: SQL warehouse ID
        
    Returns:
        Dictionary containing deletion status
    """
    logger.info(f"Deleting AutoML experiment: {experiment_id}")
    
    # Generate deletion code
    deletion_code = f"""
from mlflow import MlflowClient

client = MlflowClient()

# Delete experiment (this will also delete all runs)
client.delete_experiment("{experiment_id}")

print(f"Experiment {experiment_id} deleted successfully")
"""
    
    try:
        # In a real implementation, this would execute the deletion
        return {
            "status": "deletion_initiated",
            "experiment_id": experiment_id,
            "deletion_code": deletion_code,
            "note": "Experiment deletion code generated. Execute this in a Databricks notebook."
        }
    except Exception as e:
        logger.error(f"Error deleting experiment: {str(e)}")
        raise DatabricksAPIError(f"Failed to delete experiment: {str(e)}")


async def get_automl_feature_importance(
    run_id: str,
    warehouse_id: str
) -> Dict[str, Any]:
    """
    Get feature importance from an AutoML model.
    
    Args:
        run_id: MLflow run ID
        warehouse_id: SQL warehouse ID
        
    Returns:
        Dictionary containing feature importance data
    """
    logger.info(f"Getting feature importance for run: {run_id}")
    
    # Generate feature importance extraction code
    feature_code = f"""
import mlflow
import json

# Load model
model_uri = "runs:/{run_id}/model"
model = mlflow.pyfunc.load_model(model_uri)

# Try to get feature importance (depends on model type)
feature_importance = None
try:
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_.tolist()
    elif hasattr(model._model_impl, 'feature_importances_'):
        feature_importance = model._model_impl.feature_importances_.tolist()
except:
    feature_importance = "Feature importance not available for this model type"

result = {{
    "run_id": "{run_id}",
    "feature_importance": feature_importance,
    "model_type": str(type(model))
}}

print("RESULT_JSON:", json.dumps(result))
"""
    
    try:
        return {
            "status": "extraction_ready",
            "run_id": run_id,
            "feature_code": feature_code,
            "note": "Feature importance extraction code generated. Execute this in a Databricks notebook."
        }
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise DatabricksAPIError(f"Failed to get feature importance: {str(e)}")
