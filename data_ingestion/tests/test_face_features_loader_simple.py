import pytest
from unittest.mock import patch, Mock
import pandas as pd

class TestFaceFeaturesLoader:
    
    @patch('data_ingestion.loaders.face_features.pd.read_parquet')
    @patch('data_ingestion.loaders.face_features.BaseLoader')
    @patch('data_ingestion.loaders.face_features.get_engine')
    @patch('data_ingestion.loaders.face_features.read_from_minio')
    def test_run_success(self, mock_read_minio, mock_get_engine, mock_base_loader, mock_read_parquet):
        """Test successful loading of face features data"""
        from data_ingestion.loaders import face_features
        
        # Setup mocks
        mock_get_engine.return_value = Mock()
        mock_loader_instance = Mock()
        mock_base_loader.return_value = mock_loader_instance
        mock_read_minio.return_value = Mock()  # File-like object mock
        
        # Create mock parquet data as DataFrame
        sample_df = pd.DataFrame({
            'person_id': [1, 2, 3],
            'feature_1': [0.1, 0.2, 0.3],
            'feature_2': [0.4, 0.5, 0.6],
            'confidence': [0.95, 0.87, 0.92],
            'embedding': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]  # Include embedding column
        })
        mock_read_parquet.return_value = sample_df
        
        # Execute
        face_features.run("test_bucket", "face_features.parquet")
        
        # Verify
        mock_read_minio.assert_called_once_with("test_bucket", "face_features.parquet")
        mock_get_engine.assert_called_once()
        mock_read_parquet.assert_called_once()
        mock_loader_instance.load.assert_called_once()
    
    @patch('data_ingestion.loaders.face_features.pd.read_parquet')
    @patch('data_ingestion.loaders.face_features.BaseLoader')
    @patch('data_ingestion.loaders.face_features.get_engine')
    @patch('data_ingestion.loaders.face_features.read_from_minio')
    def test_empty_parquet_handling(self, mock_read_minio, mock_get_engine, mock_base_loader, mock_read_parquet):
        """Test handling of empty parquet files"""
        from data_ingestion.loaders import face_features
        import pandas as pd
        
        # Setup mocks
        mock_get_engine.return_value = Mock()
        mock_loader_instance = Mock()
        mock_base_loader.return_value = mock_loader_instance
        mock_read_minio.return_value = Mock()
        
        # Empty DataFrame with embedding column (so it can be dropped)
        empty_df = pd.DataFrame(columns=['embedding'])
        mock_read_parquet.return_value = empty_df
        
        # Execute - should not raise exception
        face_features.run("test_bucket", "empty_features.parquet")
        
        # Verify
        mock_loader_instance.load.assert_called_once()
    
    @patch('data_ingestion.loaders.face_features.pd.read_parquet')
    @patch('data_ingestion.loaders.face_features.BaseLoader')
    @patch('data_ingestion.loaders.face_features.get_engine')
    @patch('data_ingestion.loaders.face_features.read_from_minio')
    def test_function_calls(self, mock_read_minio, mock_get_engine, mock_base_loader, mock_read_parquet):
        """Test that all expected functions are called"""
        from data_ingestion.loaders import face_features
        import pandas as pd
        
        # Setup mocks
        mock_get_engine.return_value = Mock()
        mock_loader_instance = Mock()
        mock_base_loader.return_value = mock_loader_instance
        mock_read_minio.return_value = Mock()
        
        # Mock DataFrame
        test_df = pd.DataFrame({'test': [1, 2, 3], 'embedding': [[1], [2], [3]]})
        mock_read_parquet.return_value = test_df
        
        # Execute
        face_features.run("test_bucket", "test.parquet")
        
        # Verify all functions called correctly
        mock_read_minio.assert_called_once()
        mock_get_engine.assert_called_once()
        mock_base_loader.assert_called_once()
        mock_loader_instance.load.assert_called_once()