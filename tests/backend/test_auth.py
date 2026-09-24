def test_register_user_success(client):
    payload = {
        "email": "new_user@nexus.ai",
        "password": "securepassword123",
        "full_name": "New Scientist",
        "role": "ml_engineer"
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "new_user@nexus.ai"
    assert data["full_name"] == "New Scientist"
    assert "id" in data
    assert "hashed_password" not in data


def test_register_duplicate_email_fails(client, test_user):
    payload = {
        "email": test_user.email,
        "password": "password123",
        "full_name": "Duplicate User"
    }
    response = client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]


def test_login_success(client, test_user):
    payload = {
        "email": "test_scientist@nexus.ai",
        "password": "password123"
    }
    response = client.post("/api/v1/auth/login", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_login_invalid_password_fails(client, test_user):
    payload = {
        "email": "test_scientist@nexus.ai",
        "password": "wrongpassword"
    }
    response = client.post("/api/v1/auth/login", json=payload)
    assert response.status_code == 401


def test_get_current_user_profile(client, auth_headers, test_user):
    response = client.get("/api/v1/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == test_user.email
    assert data["id"] == test_user.id
