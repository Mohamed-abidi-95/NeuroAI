import os
import pytest

def pytest_addoption(parser):
    parser.addoption(
                   ,
        action="store",
        default=os.getenv("API_URL", "http://localhost:8000"),
        help="URL de l'API FastAPI à tester (défaut: http://localhost:8000)",
    )

@pytest.fixture(scope="session")
def api_url(request):
    url = request.config.getoption("--api-url")

    import tests.test_api as ta
    ta.API_URL = url
    return url
