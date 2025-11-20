import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app, Base, get_db

# Use a local SQLite file DB just for tests
TEST_DB_URL = "sqlite:///./test_doctor_payments.db"


engine = create_engine(
    TEST_DB_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    # Create tables once for the whole test session
    Base.metadata.create_all(bind=engine)
    yield
    # Drop tables after the session if you want a clean slate
    Base.metadata.drop_all(bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Override the dependency in the FastAPI app so tests use SQLite instead of Postgres
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture
def client():
    return TestClient(app)
