import pytest
from unittest.mock import patch, Mock
import pandas as pd

class TestZtsListLoader:
    
    @patch('data_ingestion.loaders.zts_list.pd.read_json')
    @patch('data_ingestion.loaders.zts_list.BaseLoader')
    @patch('data_ingestion.loaders.zts_list.get_engine')
    @patch('data_ingestion.loaders.zts_list.read_from_minio')
    def test_run_success(self, mock_read_minio, mock_get_engine, mock_base_loader, mock_read_json):
        """Test successful loading of ZTS list data"""
        from data_ingestion.loaders import zts_list
        
        # Setup mocks
        mock_get_engine.return_value = Mock()
        mock_loader_instance = Mock()
        mock_base_loader.return_value = mock_loader_instance
        mock_read_minio.return_value = Mock()  # File-like object
        
        # Mock JSON DataFrame result
        sample_df = pd.DataFrame([
            {"id": 1, "name": "ZTS 1", "location": "Prague"},
            {"id": 2, "name": "ZTS 2", "location": "Brno"}
        ])
        mock_read_json.return_value = sample_df
        
        # Execute
        zts_list.run("test_bucket", "zts_list.json")
        
        # Verify
        mock_read_minio.assert_called_once_with("test_bucket", "zts_list.json")
        mock_get_engine.assert_called_once()
        mock_read_json.assert_called_once()
        mock_loader_instance.load.assert_called_once()
    
    @patch('data_ingestion.loaders.zts_list.pd.read_json')
    @patch('data_ingestion.loaders.zts_list.BaseLoader')
    @patch('data_ingestion.loaders.zts_list.get_engine')
    @patch('data_ingestion.loaders.zts_list.read_from_minio')
    def test_empty_json_handling(self, mock_read_minio, mock_get_engine, mock_base_loader, mock_read_json):
        """Test handling of empty JSON files"""
        from data_ingestion.loaders import zts_list
        
        # Setup mocks
        mock_get_engine.return_value = Mock()
        mock_loader_instance = Mock()
        mock_base_loader.return_value = mock_loader_instance
        mock_read_minio.return_value = Mock()
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        mock_read_json.return_value = empty_df
        
        # Execute - should not raise exception
        zts_list.run("test_bucket", "empty_zts.json")
        
        # Verify
        mock_loader_instance.load.assert_called_once()
    
    @patch('data_ingestion.loaders.zts_list.pd.read_json')
    @patch('data_ingestion.loaders.zts_list.BaseLoader')
    @patch('data_ingestion.loaders.zts_list.get_engine')
    @patch('data_ingestion.loaders.zts_list.read_from_minio')  
    def test_malformed_json_handling(self, mock_read_minio, mock_get_engine, mock_base_loader, mock_read_json):
        """Test handling of malformed JSON"""
        from data_ingestion.loaders import zts_list
        
        # Setup mocks
        mock_get_engine.return_value = Mock()
        mock_base_loader.return_value = Mock()
        mock_read_minio.return_value = Mock()
        
        # Mock pandas JSON parsing error
        mock_read_json.side_effect = ValueError("Expected object or value")
        
        # Execute - should handle JSON parsing errors
        with pytest.raises(ValueError):
            zts_list.run("test_bucket", "malformed_zts.json")