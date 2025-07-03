#!/bin/bash

# Start MLflow MCP Server
# This script starts the MLflow MCP server with proper environment configuration

set -e

# Check if MLFLOW_TRACKING_URI is set
if [ -z "$MLFLOW_TRACKING_URI" ]; then
    echo "Warning: MLFLOW_TRACKING_URI not set. Using default: http://localhost:5000"
    export MLFLOW_TRACKING_URI="http://localhost:5000"
fi

# Set logging level if not already set
if [ -z "$LOG_LEVEL" ]; then
    export LOG_LEVEL="INFO"
fi

echo "Starting MLflow MCP Server..."
echo "MLflow Tracking URI: $MLFLOW_TRACKING_URI"
echo "Log Level: $LOG_LEVEL"

# Start the server
python -m databricks_mcp.server.mlflow_mcp_server
