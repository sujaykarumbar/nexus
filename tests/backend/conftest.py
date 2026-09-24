import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Set testing environment variables before importing app
os.environ["ENVIRONMENT"] = "testing"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from apps.api.main import app
from apps.api.core.database import Base, get_db
from apps.api.core.security import get_password_hash, create_access_token
from apps.api.models.user import User

# In-memory test engine
TEST_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh in-memory database for each test function."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session, monkeypatch):
    """Override get_db and SessionLocal with test database session."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    import apps.api.routes.jobs
    import apps.api.services.dataset_pipeline
    import apps.api.routes.ml
    import apps.api.routes.forecasting
    import apps.api.routes.anomalies
    monkeypatch.setattr(apps.api.routes.jobs, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(apps.api.services.dataset_pipeline, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(apps.api.routes.ml, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(apps.api.routes.forecasting, "SessionLocal", TestingSessionLocal)
    monkeypatch.setattr(apps.api.routes.anomalies, "SessionLocal", TestingSessionLocal)

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def test_user(db_session):
    """Create and return a default test user."""
    user = User(
        email="test_scientist@nexus.ai",
        hashed_password=get_password_hash("password123"),
        full_name="Dr. Test Scientist",
        role="data_scientist"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def auth_headers(test_user):
    """Return JWT Bearer authorization headers for test_user."""
    token = create_access_token(subject=test_user.id)
    return {"Authorization": f"Bearer {token}"}
