from pathlib import Path
import sys

root_directory = Path(__file__).resolve().parents[1]
sys.path.append(str(root_directory))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": " Hello World"}

def test_read_heathcheck():
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_wrong_router():
    response = client.get("/hello")
    assert response.status_code == 404
    assert response.json() == {"detail":"Not Found"}

