"""
API for MLflow functionality.

This module provides tools for MLflow experiment tracking, model registry,
and model serving operations using the MLflow REST API.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from databricks_mcp.core.utils import DatabricksAPIError, make_api_request

# Configure logging
logger = logging.getLogger(__name__)


# Experiment Management

async def create_experiment(
    name: str,
    artifact_location: Optional[str] = None,
    tags: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Create a new MLflow experiment.
    
    Args:
        name: Name of the experiment
        artifact_location: Optional artifact storage location
        tags: Optional list of tags as key-value pairs
        
    Returns:
        Dictionary containing experiment creation response
    """
    logger.info(f"Creating MLflow experiment: {name}")
    
    payload = {"name": name}
    if artifact_location:
        payload["artifact_location"] = artifact_location
    if tags:
        payload["tags"] = tags
    
    try:
        response = await make_api_request(
            method="POST",
            url="api/2.0/mlflow/experiments/create",
            json=payload
        )
        return response
    except Exception as e:
        logger.error(f"Error creating experiment: {str(e)}")
        raise DatabricksAPIError(f"Failed to create experiment: {str(e)}")


async def list_experiments(
    view_type: str = "ACTIVE_ONLY",
    max_results: Optional[int] = None,
    page_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    List MLflow experiments.
    
    Args:
        view_type: View type filter (ACTIVE_ONLY, DELETED_ONLY, ALL)
        max_results: Maximum number of results to return
        page_token: Token for pagination
        
    Returns:
        Dictionary containing list of experiments
    """
    logger.info("Listing MLflow experiments")
    
    params = {"view_type": view_type}
    if max_results:
        params["max_results"] = max_results
    if page_token:
        params["page_token"] = page_token
    
    try:
        response = await make_api_request(
            method="GET",
            url="api/2.0/mlflow/experiments/list",
            params=params
        )
        return response
    except Exception as e:
        logger.error(f"Error listing experiments: {str(e)}")
        raise DatabricksAPIError(f"Failed to list experiments: {str(e)}")


async def get_experiment(experiment_id: str) -> Dict[str, Any]:
    """
    Get details of a specific MLflow experiment.
    
    Args:
        experiment_id: ID of the experiment
        
    Returns:
        Dictionary containing experiment details
    """
    logger.info(f"Getting MLflow experiment: {experiment_id}")
    
    try:
        response = await make_api_request(
            method="GET",
            url="api/2.0/mlflow/experiments/get",
            params={"experiment_id": experiment_id}
        )
        return response
    except Exception as e:
        logger.error(f"Error getting experiment: {str(e)}")
        raise DatabricksAPIError(f"Failed to get experiment: {str(e)}")


async def delete_experiment(experiment_id: str) -> Dict[str, Any]:
    """
    Delete an MLflow experiment.
    
    Args:
        experiment_id: ID of the experiment to delete
        
    Returns:
        Dictionary containing deletion response
    """
    logger.info(f"Deleting MLflow experiment: {experiment_id}")
    
    try:
        response = await make_api_request(
            method="POST",
            url="api/2.0/mlflow/experiments/delete",
            json={"experiment_id": experiment_id}
        )
        return response
    except Exception as e:
        logger.error(f"Error deleting experiment: {str(e)}")
        raise DatabricksAPIError(f"Failed to delete experiment: {str(e)}")


# Run Management

async def create_run(
    experiment_id: str,
    start_time: Optional[int] = None,
    tags: Optional[List[Dict[str, str]]] = None,
    run_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new MLflow run.
    
    Args:
        experiment_id: ID of the experiment
        start_time: Start time as Unix timestamp in milliseconds
        tags: Optional list of tags as key-value pairs
        run_name: Optional name for the run
        
    Returns:
        Dictionary containing run creation response
    """
    logger.info(f"Creating MLflow run in experiment: {experiment_id}")
    
    payload = {"experiment_id": experiment_id}
    if start_time:
        payload["start_time"] = start_time
    if tags:
        payload["tags"] = tags
    if run_name:
        payload["run_name"] = run_name
    
    try:
        response = await make_api_request(
            method="POST",
            url="api/2.0/mlflow/runs/create",
            json=payload
        )
        return response
    except Exception as e:
        logger.error(f"Error creating run: {str(e)}")
        raise DatabricksAPIError(f"Failed to create run: {str(e)}")


async def get_run(run_id: str) -> Dict[str, Any]:
    """
    Get details of a specific MLflow run.
    
    Args:
        run_id: ID of the run
        
    Returns:
        Dictionary containing run details
    """
    logger.info(f"Getting MLflow run: {run_id}")
    
    try:
        response = await make_api_request(
            method="GET",
            url="api/2.0/mlflow/runs/get",
            params={"run_id": run_id}
        )
        return response
    except Exception as e:
        logger.error(f"Error getting run: {str(e)}")
        raise DatabricksAPIError(f"Failed to get run: {str(e)}")


async def log_metric(
    run_id: str,
    key: str,
    value: float,
    timestamp: Optional[int] = None,
    step: Optional[int] = None
) -> Dict[str, Any]:
    """
    Log a metric to an MLflow run.
    
    Args:
        run_id: ID of the run
        key: Metric name
        value: Metric value
        timestamp: Unix timestamp in milliseconds
        step: Step number for the metric
        
    Returns:
        Dictionary containing log response
    """
    logger.info(f"Logging metric {key}={value} to run: {run_id}")
    
    payload = {
        "run_id": run_id,
        "key": key,
        "value": value
    }
    if timestamp:
        payload["timestamp"] = timestamp
    if step is not None:
        payload["step"] = step
    
    try:
        response = await make_api_request(
            method="POST",
            url="api/2.0/mlflow/runs/log-metric",
            json=payload
        )
        return response
    except Exception as e:
        logger.error(f"Error logging metric: {str(e)}")
        raise DatabricksAPIError(f"Failed to log metric: {str(e)}")


async def log_parameter(
    run_id: str,
    key: str,
    value: str
) -> Dict[str, Any]:
    """
    Log a parameter to an MLflow run.
    
    Args:
        run_id: ID of the run
        key: Parameter name
        value: Parameter value
        
    Returns:
        Dictionary containing log response
    """
    logger.info(f"Logging parameter {key}={value} to run: {run_id}")
    
    payload = {
        "run_id": run_id,
        "key": key,
        "value": value
    }
    
    try:
        response = await make_api_request(
            method="POST",
            url="api/2.0/mlflow/runs/log-parameter",
            json=payload
        )
        return response
    except Exception as e:
        logger.error(f"Error logging parameter: {str(e)}")
        raise DatabricksAPIError(f"Failed to log parameter: {str(e)}")


async def log_batch(
    run_id: str,
    metrics: Optional[List[Dict[str, Any]]] = None,
    params: Optional[List[Dict[str, str]]] = None,
    tags: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    Log metrics, parameters, and tags in batch to an MLflow run.
    
    Args:
        run_id: ID of the run
        metrics: List of metrics to log
        params: List of parameters to log
        tags: List of tags to log
        
    Returns:
        Dictionary containing batch log response
    """
    logger.info(f"Logging batch data to run: {run_id}")
    
    payload = {"run_id": run_id}
    if metrics:
        payload["metrics"] = metrics
    if params:
        payload["params"] = params
    if tags:
        payload["tags"] = tags
    
    try:
        response = await make_api_request(
            method="POST",
            url="api/2.0/mlflow/runs/log-batch",
            json=payload
        )
        return response
    except Exception as e:
        logger.error(f"Error logging batch: {str(e)}")
        raise DatabricksAPIError(f"Failed to log batch: {str(e)}")


async def search_runs(
    experiment_ids: List[str],
    filter_string: Optional[str] = None,
    run_view_type: str = "ACTIVE_ONLY",
    max_results: Optional[int] = None,
    order_by: Optional[List[str]] = None,
    page_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Search for MLflow runs.
    
    Args:
        experiment_ids: List of experiment IDs to search
        filter_string: Filter expression
        run_view_type: View type (ACTIVE_ONLY, DELETED_ONLY, ALL)
        max_results: Maximum number of results
        order_by: List of order by clauses
        page_token: Token for pagination
        
    Returns:
        Dictionary containing search results
    """
    logger.info(f"Searching runs in experiments: {experiment_ids}")
    
    payload = {
        "experiment_ids": experiment_ids,
        "run_view_type": run_view_type
    }
    if filter_string:
        payload["filter"] = filter_string
    if max_results:
        payload["max_results"] = max_results
    if order_by:
        payload["order_by"] = order_by
    if page_token:
        payload["page_token"] = page_token
    
    try:
        response = await make_api_request(
            method="POST",
            url="api/2.0/mlflow/runs/search",
            json=payload
        )
        return response
    except Exception as e:
        logger.error(f"Error searching runs: {str(e)}")
        raise DatabricksAPIError(f"Failed to search runs: {str(e)}")


async def update_run(
    run_id: str,
    status: Optional[str] = None,
    end_time: Optional[int] = None,
    run_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update an MLflow run.
    
    Args:
        run_id: ID of the run
        status: Run status (RUNNING, SCHEDULED, FINISHED, FAILED, KILLED)
        end_time: End time as Unix timestamp in milliseconds
        run_name: New name for the run
        
    Returns:
        Dictionary containing update response
    """
    logger.info(f"Updating MLflow run: {run_id}")
    
    payload = {"run_id": run_id}
    if status:
        payload["status"] = status
    if end_time:
        payload["end_time"] = end_time
    if run_name:
        payload["run_name"] = run_name
    
    try:
        response = await make_api_request(
            method="POST",
            url="api/2.0/mlflow/runs/update",
            json=payload
        )
        return response
    except Exception as e:
        logger.error(f"Error updating run: {str(e)}")
        raise DatabricksAPIError(f"Failed to update run: {str(e)}")


# Model Registry

async def create_registered_model(
    name: str,
    tags: Optional[List[Dict[str, str]]] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Register a new model in MLflow Model Registry.
    
    Args:
        name: Name of the model
        tags: Optional list of tags
        description: Optional description
        
    Returns:
        Dictionary containing model registration response
    """
    logger.info(f"Creating registered model: {name}")
    
    payload = {"name": name}
    if tags:
        payload["tags"] = tags
    if description:
        payload["description"] = description
    
    try:
        response = await make_api_request(
            method="POST",
            url="api/2.0/mlflow/registered-models/create",
            json=payload
        )
        return response
    except Exception as e:
        logger.error(f"Error creating registered model: {str(e)}")
        raise DatabricksAPIError(f"Failed to create registered model: {str(e)}")


async def list_registered_models(
    max_results: Optional[int] = None,
    page_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    List all registered models.
    
    Args:
        max_results: Maximum number of results
        page_token: Token for pagination
        
    Returns:
        Dictionary containing list of registered models
    """
    logger.info("Listing registered models")
    
    params = {}
    if max_results:
        params["max_results"] = max_results
    if page_token:
        params["page_token"] = page_token
    
    try:
        response = await make_api_request(
            method="GET",
            url="api/2.0/mlflow/registered-models/list",
            params=params
        )
        return response
    except Exception as e:
        logger.error(f"Error listing registered models: {str(e)}")
        raise DatabricksAPIError(f"Failed to list registered models: {str(e)}")


async def get_registered_model(name: str) -> Dict[str, Any]:
    """
    Get details of a registered model.
    
    Args:
        name: Name of the model
        
    Returns:
        Dictionary containing model details
    """
    logger.info(f"Getting registered model: {name}")
    
    try:
        response = await make_api_request(
            method="GET",
            url="api/2.0/mlflow/registered-models/get",
            params={"name": name}
        )
        return response
    except Exception as e:
        logger.error(f"Error getting registered model: {str(e)}")
        raise DatabricksAPIError(f"Failed to get registered model: {str(e)}")


async def create_model_version(
    name: str,
    source: str,
    run_id: Optional[str] = None,
    tags: Optional[List[Dict[str, str]]] = None,
    run_link: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new version of a registered model.
    
    Args:
        name: Name of the model
        source: Source path of the model artifacts
        run_id: MLflow run ID that generated this model
        tags: Optional list of tags
        run_link: Optional link to the run
        description: Optional description
        
    Returns:
        Dictionary containing model version creation response
    """
    logger.info(f"Creating model version for: {name}")
    
    payload = {
        "name": name,
        "source": source
    }
    if run_id:
        payload["run_id"] = run_id
    if tags:
        payload["tags"] = tags
    if run_link:
        payload["run_link"] = run_link
    if description:
        payload["description"] = description
    
    try:
        response = await make_api_request(
            method="POST",
            url="api/2.0/mlflow/model-versions/create",
            json=payload
        )
        return response
    except Exception as e:
        logger.error(f"Error creating model version: {str(e)}")
        raise DatabricksAPIError(f"Failed to create model version: {str(e)}")


async def get_model_version(name: str, version: str) -> Dict[str, Any]:
    """
    Get details of a specific model version.
    
    Args:
        name: Name of the model
        version: Version number
        
    Returns:
        Dictionary containing model version details
    """
    logger.info(f"Getting model version: {name} v{version}")
    
    try:
        response = await make_api_request(
            method="GET",
            url="api/2.0/mlflow/model-versions/get",
            params={"name": name, "version": version}
        )
        return response
    except Exception as e:
        logger.error(f"Error getting model version: {str(e)}")
        raise DatabricksAPIError(f"Failed to get model version: {str(e)}")


async def transition_model_version_stage(
    name: str,
    version: str,
    stage: str,
    archive_existing_versions: bool = False
) -> Dict[str, Any]:
    """
    Transition a model version to a different stage.
    
    Args:
        name: Name of the model
        version: Version number
        stage: Target stage (None, Staging, Production, Archived)
        archive_existing_versions: Whether to archive existing versions in the target stage
        
    Returns:
        Dictionary containing transition response
    """
    logger.info(f"Transitioning model {name} v{version} to stage: {stage}")
    
    payload = {
        "name": name,
        "version": version,
        "stage": stage,
        "archive_existing_versions": archive_existing_versions
    }
    
    try:
        response = await make_api_request(
            method="POST",
            url="api/2.0/mlflow/model-versions/transition-stage",
            json=payload
        )
        return response
    except Exception as e:
        logger.error(f"Error transitioning model version stage: {str(e)}")
        raise DatabricksAPIError(f"Failed to transition model version stage: {str(e)}")


async def delete_model_version(name: str, version: str) -> Dict[str, Any]:
    """
    Delete a model version.
    
    Args:
        name: Name of the model
        version: Version number
        
    Returns:
        Dictionary containing deletion response
    """
    logger.info(f"Deleting model version: {name} v{version}")
    
    try:
        response = await make_api_request(
            method="DELETE",
            url="api/2.0/mlflow/model-versions/delete",
            json={"name": name, "version": version}
        )
        return response
    except Exception as e:
        logger.error(f"Error deleting model version: {str(e)}")
        raise DatabricksAPIError(f"Failed to delete model version: {str(e)}")


async def get_latest_versions(
    name: str,
    stages: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Get the latest versions of a model for specified stages.
    
    Args:
        name: Name of the model
        stages: List of stages to get latest versions for
        
    Returns:
        Dictionary containing latest model versions
    """
    logger.info(f"Getting latest versions for model: {name}")
    
    payload = {"name": name}
    if stages:
        payload["stages"] = stages
    
    try:
        response = await make_api_request(
            method="POST",
            url="api/2.0/mlflow/registered-models/get-latest-versions",
            json=payload
        )
        return response
    except Exception as e:
        logger.error(f"Error getting latest versions: {str(e)}")
        raise DatabricksAPIError(f"Failed to get latest versions: {str(e)}")


# Model Serving and Predictions

async def create_model_serving_endpoint(
    name: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a model serving endpoint.
    
    Args:
        name: Name of the serving endpoint
        config: Endpoint configuration
        
    Returns:
        Dictionary containing endpoint creation response
    """
    logger.info(f"Creating model serving endpoint: {name}")
    
    payload = {
        "name": name,
        "config": config
    }
    
    try:
        response = await make_api_request(
            method="POST",
            url="api/2.0/serving-endpoints",
            json=payload
        )
        return response
    except Exception as e:
        logger.error(f"Error creating serving endpoint: {str(e)}")
        raise DatabricksAPIError(f"Failed to create serving endpoint: {str(e)}")


async def list_serving_endpoints() -> Dict[str, Any]:
    """
    List all model serving endpoints.
    
    Returns:
        Dictionary containing list of serving endpoints
    """
    logger.info("Listing model serving endpoints")
    
    try:
        response = await make_api_request(
            method="GET",
            url="api/2.0/serving-endpoints"
        )
        return response
    except Exception as e:
        logger.error(f"Error listing serving endpoints: {str(e)}")
        raise DatabricksAPIError(f"Failed to list serving endpoints: {str(e)}")


async def get_serving_endpoint(name: str) -> Dict[str, Any]:
    """
    Get details of a serving endpoint.
    
    Args:
        name: Name of the serving endpoint
        
    Returns:
        Dictionary containing endpoint details
    """
    logger.info(f"Getting serving endpoint: {name}")
    
    try:
        response = await make_api_request(
            method="GET",
            url=f"api/2.0/serving-endpoints/{name}"
        )
        return response
    except Exception as e:
        logger.error(f"Error getting serving endpoint: {str(e)}")
        raise DatabricksAPIError(f"Failed to get serving endpoint: {str(e)}")


async def predict_model(
    endpoint_name: str,
    inputs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Make predictions using a model serving endpoint.
    
    Args:
        endpoint_name: Name of the serving endpoint
        inputs: Input data for prediction
        
    Returns:
        Dictionary containing prediction results
    """
    logger.info(f"Making prediction with endpoint: {endpoint_name}")
    
    try:
        response = await make_api_request(
            method="POST",
            url=f"serving-endpoints/{endpoint_name}/invocations",
            json=inputs
        )
        return response
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise DatabricksAPIError(f"Failed to make prediction: {str(e)}")
