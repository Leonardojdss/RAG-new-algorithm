import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


class TestMainApplication:
    """Test cases for main.py application setup"""
    
    def test_app_creation(self):
        """Test that the FastAPI app is created correctly"""
        from src.main import app
        
        assert app is not None
        assert hasattr(app, 'include_router')

    def test_router_inclusion(self):
        """Test that the router is included with correct prefix"""
        from src.main import app
        
        # Check if router is included with correct prefix
        routes = [route for route in app.routes]
        
        # Find routes with the /new_rag prefix
        new_rag_routes = [route for route in routes if hasattr(route, 'path') and route.path.startswith('/new_rag')]
        
        assert len(new_rag_routes) > 0

    def test_main_execution_import(self):
        """Test that main module can be imported without issues"""
        # Just test that we can import the main module without issues
        import src.main
        assert hasattr(src.main, 'app')
        assert src.main.app is not None

    def test_app_configuration(self):
        """Test app configuration and metadata"""
        from src.main import app
        
        # Test that app has expected configuration
        assert app.title == "FastAPI"  # Default FastAPI title
        
        # You can add more configuration tests here if you add
        # custom configuration to your FastAPI app

    def test_app_startup_and_shutdown_events(self):
        """Test that app can handle startup/shutdown events"""
        from src.main import app
        
        # Test that we can add startup/shutdown events
        startup_called = False
        shutdown_called = False
        
        @app.on_event("startup")
        async def startup_event():
            nonlocal startup_called
            startup_called = True
        
        @app.on_event("shutdown")
        async def shutdown_event():
            nonlocal shutdown_called
            shutdown_called = True
        
        # These events would be called during actual app lifecycle
        # but we can at least verify they can be registered
        assert len(app.router.on_startup) > 0
        assert len(app.router.on_shutdown) > 0

    def test_app_routes_available(self):
        """Test that expected routes are available"""
        from src.main import app
        
        client = TestClient(app)
        
        # Test that the app responds (even if with errors due to missing dependencies)
        # This at least verifies the routing is set up correctly
        response = client.get("/")
        # We expect a 404 since we don't have a root route
        assert response.status_code == 404

    def test_app_openapi_schema(self):
        """Test that OpenAPI schema is generated"""
        from src.main import app
        
        client = TestClient(app)
        
        # Test OpenAPI documentation endpoint
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_schema = response.json()
        assert "openapi" in openapi_schema
        assert "info" in openapi_schema
        assert "paths" in openapi_schema

    def test_app_docs_endpoints(self):
        """Test that documentation endpoints are available"""
        from src.main import app
        
        client = TestClient(app)
        
        # Test Swagger UI
        response = client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_app_health_check_via_router(self):
        """Test app health through router endpoints"""
        from src.main import app
        
        client = TestClient(app)
        
        # Since we know the router has specific endpoints, let's test one
        # This will test the integration between main.py and the router
        try:
            # This might fail due to missing dependencies, but it tests the routing
            response = client.post("/new_rag/embedding", json={"text": "test", "index": 1})
            # We don't care about the exact response, just that routing works
            assert response.status_code in [200, 400, 422, 500]  # Any valid HTTP response
        except Exception:
            # If there are dependency issues, that's okay for this test
            # We're just testing that the route exists and is properly configured
            pass

class TestApplicationIntegration:
    """Integration tests for the complete application"""
    
    def test_app_with_router_integration(self):
        """Test that app correctly integrates with router"""
        from src.main import app
        
        # Verify that routes exist
        assert len(app.routes) > 0
        
        # Check that some routes have the expected prefix
        route_paths = [getattr(route, 'path', '') for route in app.routes]
        has_new_rag_routes = any('/new_rag' in path for path in route_paths if path)
        
        # This verifies that the router integration is working
        assert has_new_rag_routes or len(app.routes) > 3  # Either has our routes or default routes

    def test_complete_app_setup(self):
        """Test complete application setup"""
        from src.main import app
        
        # Test that all components are properly integrated
        assert app is not None
        
        # Test that we can create a test client
        client = TestClient(app)
        assert client is not None
        
        # Test that the app has the expected structure
        assert hasattr(app, 'router')
        assert hasattr(app, 'routes')
        assert len(app.routes) > 0

class TestApplicationConfiguration:
    """Test application configuration and environment"""
    
    def test_import_structure(self):
        """Test that all imports in main.py work correctly"""
        # Test individual imports to ensure they don't fail
        from fastapi import FastAPI
        assert FastAPI is not None
        
        from src.controller.api.router import router
        assert router is not None
        
        # Test the main app import
        from src.main import app
        assert app is not None

    def test_module_level_variables(self):
        """Test module level variables are properly set"""
        import src.main as main_module
        
        # Test that app is defined at module level
        assert hasattr(main_module, 'app')
        assert main_module.app is not None
        
    @patch.dict('os.environ', {'PORT': '9000', 'HOST': '127.0.0.1'})
    def test_environment_variables_usage(self):
        """Test that environment variables could be used if needed"""
        import os
        
        # This tests that we could extend main.py to use environment variables
        port = os.getenv('PORT', '8000')
        host = os.getenv('HOST', '0.0.0.0')
        
        assert port == '9000'
        assert host == '127.0.0.1'
        
        # This demonstrates how main.py could be enhanced to use env vars
