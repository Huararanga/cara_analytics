import pytest
from unittest.mock import Mock
import tempfile
import os

# Delay pandas import to avoid immediate failure if not installed
def get_pandas():
    try:
        import pandas as pd
        return pd
    except ImportError:
        pytest.skip("pandas not available")

@pytest.fixture  
def mock_engine():
    """Create an in-memory SQLite engine for testing"""
    try:
        from sqlalchemy import create_engine
        engine = create_engine("sqlite:///:memory:")
        return engine
    except ImportError:
        pytest.skip("sqlalchemy not available")

@pytest.fixture
def sqlite_base_loader():
    """Patch BaseLoader to work with SQLite (no schema support)"""
    def _load_for_sqlite(self, df):
        # Simple SQLite-compatible load method
        df.to_sql(
            name=self.model_class.__tablename__,
            con=self.engine,
            if_exists='replace',  # Replace instead of drop/create
            index=False,
            method='multi',
            chunksize=1000
        )
    
    # Return the patched method
    return _load_for_sqlite

@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing file loading"""
    mock_client = Mock()
    return mock_client

@pytest.fixture
def sample_czech_cities_data():
    """Sample Czech cities data for testing"""
    pd = get_pandas()
    return pd.DataFrame({
        "country": ["CZ", "CZ", "CZ"],
        "postal_code": ["110 00", "120 00", "130 00"],
        "place_name": ["Praha", "Brno", "Ostrava"],
        "admin_name1": ["Praha", "Jihomoravský", "Moravskoslezský"],
        "admin_code1": [1, 2, 3],
        "admin_name2": ["Praha", "Brno-město", "Ostrava-město"],
        "admin_code2": [11, 22, 33],
        "admin_name3": [None, None, None],
        "admin_code3": [None, None, None],
        "latitude": [50.0755, 49.1951, 49.8209],
        "longitude": [14.4378, 16.6068, 18.2625],
        "accuracy": [4, 4, 4]
    })

@pytest.fixture
def sample_population_density_data():
    """Sample population density data for testing"""
    pd = get_pandas()
    return pd.DataFrame({
        "X": [14.2945, 14.3029, 14.4945],
        "Y": [51.0537, 51.0537, 51.0537],
        "Z": [25.44, 16.38, 416.28]
    })

@pytest.fixture
def sample_face_features_data():
    """Sample face features data for testing"""
    pd = get_pandas()
    return pd.DataFrame({
        "person_id": [1, 2, 3],
        "feature_vector": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        "confidence": [0.95, 0.87, 0.92]
    })

@pytest.fixture
def sample_zts_list_data():
    """Sample ZTS list data for testing"""
    return [
        {"id": 1, "name": "ZTS 1", "location": "Prague"},
        {"id": 2, "name": "ZTS 2", "location": "Brno"},
        {"id": 3, "name": "ZTS 3", "location": "Ostrava"}
    ]

@pytest.fixture
def temp_csv_file():
    """Create temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("X,Y,Z\n14.2945,51.0537,25.44\n14.3029,51.0537,16.38\n")
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    os.unlink(temp_file)