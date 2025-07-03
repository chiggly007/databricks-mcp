"""
MLflow MCP Server

This module implements a standalone MCP server that provides tools for interacting
with MLflow APIs. It follows the Model Context Protocol standard, communicating
via stdio and connecting to MLflow tracking servers when tools are invoked.
"""

import asyncio
import json
import logging
import sys
import os
from typing import Any, Dict, List, Optional, Union, cast

from mcp.server import FastMCP
from mcp.types import TextContent
from mcp.server.stdio import stdio_server

from databricks_mcp.api import mlflow
from databricks_mcp.core.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    filename="mlflow_mcp.log",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _unwrap_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unwrap parameters from MCP client structure.
    
    MCP clients may pass parameters in nested structure like:
    {"params": {"actual_parameter": "value"}}
    
    This function handles both nested and flat parameter structures.
    
    Args:
        params: Parameters from MCP client
        
    Returns:
        Unwrapped parameters dictionary
    """
    if 'params' in params and isinstance(params['params'], dict):
        return params['params']
    return params


class MLflowMCPServer(FastMCP):
    """An MCP server for MLflow APIs."""

    def __init__(self):
        """Initialize the MLflow MCP server."""
        super().__init__(name="mlflow-mcp", 
                         version="1.0.0", 
                         instructions="Use this server to manage MLflow experiments, runs, and models")
        logger.info("Initializing MLflow MCP server")
        logger.info(f"MLflow tracking URI: {os.getenv('MLFLOW_TRACKING_URI', 'Not set')}")
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register all MLflow MCP tools."""
        
        # Experiment management tools
        @self.tool(
            name="create_experiment",
            description="Create a new MLflow experiment with parameters: name (required), artifact_location (optional), tags (optional)"
        )
        async def create_experiment_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Creating experiment with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.create_experiment(
                    name=actual_params.get("name"),
                    artifact_location=actual_params.get("artifact_location"),
                    tags=actual_params.get("tags")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error creating experiment: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="list_experiments",
            description="List MLflow experiments with parameters: view_type (optional: ACTIVE_ONLY, DELETED_ONLY, ALL), max_results (optional), page_token (optional)"
        )
        async def list_experiments_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Listing experiments with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.list_experiments(
                    view_type=actual_params.get("view_type", "ACTIVE_ONLY"),
                    max_results=actual_params.get("max_results"),
                    page_token=actual_params.get("page_token")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error listing experiments: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="get_experiment",
            description="Get details of a specific MLflow experiment with parameter: experiment_id (required)"
        )
        async def get_experiment_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Getting experiment with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.get_experiment(actual_params.get("experiment_id"))
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error getting experiment: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="delete_experiment",
            description="Delete an MLflow experiment with parameter: experiment_id (required)"
        )
        async def delete_experiment_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Deleting experiment with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.delete_experiment(actual_params.get("experiment_id"))
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error deleting experiment: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        # Run management tools
        @self.tool(
            name="create_run",
            description="Create a new MLflow run with parameters: experiment_id (required), start_time (optional), tags (optional), run_name (optional)"
        )
        async def create_run_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Creating run with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.create_run(
                    experiment_id=actual_params.get("experiment_id"),
                    start_time=actual_params.get("start_time"),
                    tags=actual_params.get("tags"),
                    run_name=actual_params.get("run_name")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error creating run: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="get_run",
            description="Get details of a specific MLflow run with parameter: run_id (required)"
        )
        async def get_run_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Getting run with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.get_run(actual_params.get("run_id"))
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error getting run: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="log_metric",
            description="Log a metric to an MLflow run with parameters: run_id (required), key (required), value (required), timestamp (optional), step (optional)"
        )
        async def log_metric_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Logging metric with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.log_metric(
                    run_id=actual_params.get("run_id"),
                    key=actual_params.get("key"),
                    value=actual_params.get("value"),
                    timestamp=actual_params.get("timestamp"),
                    step=actual_params.get("step")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error logging metric: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="log_parameter",
            description="Log a parameter to an MLflow run with parameters: run_id (required), key (required), value (required)"
        )
        async def log_parameter_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Logging parameter with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.log_parameter(
                    run_id=actual_params.get("run_id"),
                    key=actual_params.get("key"),
                    value=actual_params.get("value")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error logging parameter: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="log_batch",
            description="Log metrics, parameters, and tags in batch to an MLflow run with parameters: run_id (required), metrics (optional), params (optional), tags (optional)"
        )
        async def log_batch_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Logging batch data with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.log_batch(
                    run_id=actual_params.get("run_id"),
                    metrics=actual_params.get("metrics"),
                    params=actual_params.get("params"),
                    tags=actual_params.get("tags")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error logging batch: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="search_runs",
            description="Search for MLflow runs with parameters: experiment_ids (required), filter_string (optional), run_view_type (optional), max_results (optional), order_by (optional), page_token (optional)"
        )
        async def search_runs_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Searching runs with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.search_runs(
                    experiment_ids=actual_params.get("experiment_ids"),
                    filter_string=actual_params.get("filter_string"),
                    run_view_type=actual_params.get("run_view_type", "ACTIVE_ONLY"),
                    max_results=actual_params.get("max_results"),
                    order_by=actual_params.get("order_by"),
                    page_token=actual_params.get("page_token")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error searching runs: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="update_run",
            description="Update an MLflow run with parameters: run_id (required), status (optional), end_time (optional), run_name (optional)"
        )
        async def update_run_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Updating run with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.update_run(
                    run_id=actual_params.get("run_id"),
                    status=actual_params.get("status"),
                    end_time=actual_params.get("end_time"),
                    run_name=actual_params.get("run_name")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error updating run: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        # Model Registry tools
        @self.tool(
            name="create_registered_model",
            description="Register a new model in MLflow Model Registry with parameters: name (required), tags (optional), description (optional)"
        )
        async def create_registered_model_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Creating registered model with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.create_registered_model(
                    name=actual_params.get("name"),
                    tags=actual_params.get("tags"),
                    description=actual_params.get("description")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error creating registered model: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="list_registered_models",
            description="List all registered models with parameters: max_results (optional), page_token (optional)"
        )
        async def list_registered_models_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Listing registered models with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.list_registered_models(
                    max_results=actual_params.get("max_results"),
                    page_token=actual_params.get("page_token")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error listing registered models: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="get_registered_model",
            description="Get details of a registered model with parameter: name (required)"
        )
        async def get_registered_model_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Getting registered model with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.get_registered_model(actual_params.get("name"))
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error getting registered model: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="create_model_version",
            description="Create a new version of a registered model with parameters: name (required), source (required), run_id (optional), tags (optional), run_link (optional), description (optional)"
        )
        async def create_model_version_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Creating model version with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.create_model_version(
                    name=actual_params.get("name"),
                    source=actual_params.get("source"),
                    run_id=actual_params.get("run_id"),
                    tags=actual_params.get("tags"),
                    run_link=actual_params.get("run_link"),
                    description=actual_params.get("description")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error creating model version: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="get_model_version",
            description="Get details of a specific model version with parameters: name (required), version (required)"
        )
        async def get_model_version_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Getting model version with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.get_model_version(
                    name=actual_params.get("name"),
                    version=actual_params.get("version")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error getting model version: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="transition_model_version_stage",
            description="Transition a model version to a different stage with parameters: name (required), version (required), stage (required: None/Staging/Production/Archived), archive_existing_versions (optional)"
        )
        async def transition_model_version_stage_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Transitioning model version stage with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.transition_model_version_stage(
                    name=actual_params.get("name"),
                    version=actual_params.get("version"),
                    stage=actual_params.get("stage"),
                    archive_existing_versions=actual_params.get("archive_existing_versions", False)
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error transitioning model version stage: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="delete_model_version",
            description="Delete a model version with parameters: name (required), version (required)"
        )
        async def delete_model_version_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Deleting model version with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.delete_model_version(
                    name=actual_params.get("name"),
                    version=actual_params.get("version")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error deleting model version: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="get_latest_versions",
            description="Get the latest versions of a model for specified stages with parameters: name (required), stages (optional)"
        )
        async def get_latest_versions_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Getting latest versions with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.get_latest_versions(
                    name=actual_params.get("name"),
                    stages=actual_params.get("stages")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error getting latest versions: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        # Model Serving tools
        @self.tool(
            name="create_model_serving_endpoint",
            description="Create a model serving endpoint with parameters: name (required), config (required)"
        )
        async def create_model_serving_endpoint_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Creating model serving endpoint with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.create_model_serving_endpoint(
                    name=actual_params.get("name"),
                    config=actual_params.get("config")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error creating model serving endpoint: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="list_serving_endpoints",
            description="List all model serving endpoints"
        )
        async def list_serving_endpoints_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info("Listing model serving endpoints")
            try:
                result = await mlflow.list_serving_endpoints()
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error listing serving endpoints: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="get_serving_endpoint",
            description="Get details of a serving endpoint with parameter: name (required)"
        )
        async def get_serving_endpoint_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Getting serving endpoint with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.get_serving_endpoint(actual_params.get("name"))
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error getting serving endpoint: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]

        @self.tool(
            name="predict_model",
            description="Make predictions using a model serving endpoint with parameters: endpoint_name (required), inputs (required)"
        )
        async def predict_model_tool(params: Dict[str, Any]) -> List[TextContent]:
            logger.info(f"Making prediction with params: {params}")
            try:
                actual_params = _unwrap_params(params)
                result = await mlflow.predict_model(
                    endpoint_name=actual_params.get("endpoint_name"),
                    inputs=actual_params.get("inputs")
                )
                return [{"type": "text", "text": json.dumps(result)}]
            except Exception as e:
                logger.error(f"Error making prediction: {str(e)}")
                return [{"type": "text", "text": json.dumps({"error": str(e)})}]


def main():
    """Main entry point for the MLflow MCP server."""
    try:
        logger.info("Starting MLflow MCP server")
        
        # Turn off buffering in stdout
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(line_buffering=True)
        
        server = MLflowMCPServer()
        
        # Use the FastMCP run method which handles async internally
        server.run()
            
    except Exception as e:
        logger.error(f"Error in MLflow MCP server: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
