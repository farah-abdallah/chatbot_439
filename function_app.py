import azure.functions as func
import logging
import sys
import json
from pathlib import Path
from io import BytesIO

# Add the project directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Import your FastAPI app
from main import app as fastapi_app
from fastapi.testclient import TestClient

# Create test client for FastAPI
client = TestClient(fastapi_app)

# Create Azure Function App
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.function_name(name="chatbot")
@app.route(route="{*route}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
async def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function that wraps the FastAPI application using TestClient.
    """
    logging.info(f'Processing: {req.method} {req.url}')
    
    try:
        # Extract path from Azure Functions route
        route_params = req.route_params.get('route', '')
        path = f"/{route_params}" if route_params else "/"
        
        # Add query parameters
        if req.params:
            query_string = "&".join([f"{k}={v}" for k, v in req.params.items()])
            path = f"{path}?{query_string}"
        
        logging.info(f"Routing to path: {path}")
        
        # Prepare request data
        headers = dict(req.headers)
        
        # Handle different HTTP methods
        if req.method == "GET":
            response = client.get(path, headers=headers)
        elif req.method == "POST":
            # Handle both JSON and form data
            content_type = req.headers.get('content-type', '')
            
            if 'application/json' in content_type:
                body = req.get_json()
                response = client.post(path, json=body, headers=headers)
            elif 'multipart/form-data' in content_type:
                # Handle file uploads
                files = {}
                for key in req.files.keys():
                    file = req.files[key]
                    files[key] = (file.filename, BytesIO(file.read()), file.content_type)
                response = client.post(path, files=files, headers=headers)
            else:
                body = req.get_body()
                response = client.post(path, data=body, headers=headers)
        elif req.method == "PUT":
            body = req.get_json() if req.get_body() else None
            response = client.put(path, json=body, headers=headers)
        elif req.method == "DELETE":
            response = client.delete(path, headers=headers)
        elif req.method == "OPTIONS":
            response = client.options(path, headers=headers)
        else:
            return func.HttpResponse(
                f"Method {req.method} not supported",
                status_code=405
            )
        
        # Convert FastAPI response to Azure Functions response
        return func.HttpResponse(
            body=response.content,
            status_code=response.status_code,
            headers=dict(response.headers),
            mimetype=response.headers.get('content-type', 'application/json')
        )
        
    except Exception as e:
        logging.error(f"Error: {str(e)}", exc_info=True)
        return func.HttpResponse(
            f"Error: {str(e)}",
            status_code=500
        )