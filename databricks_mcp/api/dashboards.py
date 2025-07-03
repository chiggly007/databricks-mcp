"""
API for managing Databricks Lakeview (AI/BI) Dashboards.
"""

import logging
from typing import Any, Dict, List, Optional

from databricks_mcp.core.utils import DatabricksAPIError, make_api_request

# Configure logging
logger = logging.getLogger(__name__)


async def create_dashboard(
    name: str,
    description: Optional[str] = None,
    widgets: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Create a new draft dashboard.
    
    Args:
        name: Name of the dashboard
        description: Optional description of the dashboard
        widgets: Optional list of widget configurations
        **kwargs: Additional dashboard configuration parameters
        
    Returns:
        Response containing the created dashboard information
        
    Raises:
        DatabricksAPIError: If the API request fails
    """
    logger.info(f"Creating dashboard: {name}")
    
    payload = {"name": name}
    if description:
        payload["description"] = description
    if widgets:
        payload["widgets"] = widgets
    
    # Add any additional configuration parameters
    payload.update(kwargs)
    
    return await make_api_request("POST", "/api/2.0/lakeview/dashboards", data=payload)


async def get_dashboard(dashboard_id: str) -> Dict[str, Any]:
    """
    Retrieve a specific dashboard (draft or published).
    
    Args:
        dashboard_id: ID of the dashboard to retrieve
        
    Returns:
        Response containing dashboard information
        
    Raises:
        DatabricksAPIError: If the API request fails
    """
    logger.info(f"Getting dashboard: {dashboard_id}")
    return await make_api_request("GET", f"/api/2.0/lakeview/dashboards/{dashboard_id}")


async def list_dashboards() -> Dict[str, Any]:
    """
    List all dashboards.
    
    Returns:
        Response containing a list of all dashboards
        
    Raises:
        DatabricksAPIError: If the API request fails
    """
    logger.info("Listing all dashboards")
    return await make_api_request("GET", "/api/2.0/lakeview/dashboards")


async def update_dashboard(
    dashboard_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    widgets: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Update an existing draft dashboard.
    
    Args:
        dashboard_id: ID of the dashboard to update
        name: Optional new name for the dashboard
        description: Optional new description for the dashboard
        widgets: Optional updated list of widget configurations
        **kwargs: Additional dashboard configuration parameters to update
        
    Returns:
        Response containing the updated dashboard information
        
    Raises:
        DatabricksAPIError: If the API request fails
    """
    logger.info(f"Updating dashboard: {dashboard_id}")
    
    payload = {}
    if name:
        payload["name"] = name
    if description:
        payload["description"] = description
    if widgets:
        payload["widgets"] = widgets
    
    # Add any additional configuration parameters
    payload.update(kwargs)
    
    return await make_api_request("PATCH", f"/api/2.0/lakeview/dashboards/{dashboard_id}", data=payload)


async def delete_dashboard(dashboard_id: str, permanent: bool = False, path: Optional[str] = None) -> Dict[str, Any]:
    """
    Delete a dashboard. Can be soft delete (trash) or permanent delete.
    
    Args:
        dashboard_id: ID of the dashboard to delete
        permanent: If True, permanently delete the dashboard using Workspace API
        path: Required for permanent deletion - the workspace path to the dashboard file
        
    Returns:
        Empty response on success
        
    Raises:
        DatabricksAPIError: If the API request fails
        ValueError: If permanent=True but path is not provided
    """
    if permanent:
        if not path:
            raise ValueError("path parameter is required for permanent deletion")
        logger.info(f"Permanently deleting dashboard file: {path}")
        return await make_api_request("POST", "/api/2.0/workspace/delete", data={"path": path})
    else:
        logger.info(f"Moving dashboard to trash: {dashboard_id}")
        return await make_api_request("DELETE", f"/api/2.0/lakeview/dashboards/{dashboard_id}")


async def publish_dashboard(dashboard_id: str) -> Dict[str, Any]:
    """
    Publish a draft dashboard.
    
    Args:
        dashboard_id: ID of the dashboard to publish
        
    Returns:
        Response containing the published dashboard information
        
    Raises:
        DatabricksAPIError: If the API request fails
    """
    logger.info(f"Publishing dashboard: {dashboard_id}")
    return await make_api_request("POST", f"/api/2.0/lakeview/dashboards/{dashboard_id}/published")


async def unpublish_dashboard(dashboard_id: str) -> Dict[str, Any]:
    """
    Unpublish a dashboard (removes published version, retains draft).
    
    Args:
        dashboard_id: ID of the dashboard to unpublish
        
    Returns:
        Empty response on success
        
    Raises:
        DatabricksAPIError: If the API request fails
    """
    logger.info(f"Unpublishing dashboard: {dashboard_id}")
    return await make_api_request("DELETE", f"/api/2.0/lakeview/dashboards/{dashboard_id}/published")


async def get_dashboard_status(dashboard_id: str) -> Dict[str, Any]:
    """
    Get concise status information for a dashboard.
    
    Args:
        dashboard_id: ID of the dashboard
        
    Returns:
        Dictionary containing dashboard status information
        
    Raises:
        DatabricksAPIError: If the API request fails
    """
    logger.info(f"Getting status for dashboard: {dashboard_id}")
    dashboard_info = await get_dashboard(dashboard_id)
    
    return {
        "dashboard_id": dashboard_id,
        "name": dashboard_info.get("name"),
        "description": dashboard_info.get("description"),
        "state": dashboard_info.get("state"),
        "created_time": dashboard_info.get("created_time"),
        "updated_time": dashboard_info.get("updated_time"),
        "is_published": dashboard_info.get("is_published", False),
    }


async def list_dashboard_widgets(dashboard_id: str) -> Dict[str, Any]:
    """
    Get the widgets configuration for a specific dashboard.
    
    Args:
        dashboard_id: ID of the dashboard
        
    Returns:
        Dictionary containing widgets information
        
    Raises:
        DatabricksAPIError: If the API request fails
    """
    logger.info(f"Getting widgets for dashboard: {dashboard_id}")
    dashboard_info = await get_dashboard(dashboard_id)
    
    return {
        "dashboard_id": dashboard_id,
        "widgets": dashboard_info.get("widgets", []),
        "widget_count": len(dashboard_info.get("widgets", [])),
    }


async def clone_dashboard(
    source_dashboard_id: str,
    new_name: str,
    new_description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Clone an existing dashboard by creating a new one with the same configuration.
    
    Args:
        source_dashboard_id: ID of the dashboard to clone
        new_name: Name for the new cloned dashboard
        new_description: Optional description for the new dashboard
        
    Returns:
        Response containing the newly created dashboard information
        
    Raises:
        DatabricksAPIError: If the API request fails
    """
    logger.info(f"Cloning dashboard {source_dashboard_id} as '{new_name}'")
    
    # Get the source dashboard configuration
    source_dashboard = await get_dashboard(source_dashboard_id)
    
    # Create a new dashboard with the same configuration
    clone_config = {
        "name": new_name,
        "description": new_description or source_dashboard.get("description"),
        "widgets": source_dashboard.get("widgets", []),
    }
    
    # Copy other relevant configuration from source
    for key in ["layout", "parameters", "settings"]:
        if key in source_dashboard:
            clone_config[key] = source_dashboard[key]
    
    return await create_dashboard(**clone_config)
