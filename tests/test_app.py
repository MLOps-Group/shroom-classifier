from shroom_classifier.app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "OK", "status-code": 200}


if __name__ == "__main__":
    test_root()
