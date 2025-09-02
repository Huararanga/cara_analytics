import pytest
from unittest.mock import patch, Mock
import pandas as pd

class TestPersonRecommendationLoader:
    
    @patch('data_ingestion.loaders.person_recommendation.BaseLoader')
    @patch('data_ingestion.loaders.person_recommendation.pd.read_sql')
    @patch('data_ingestion.loaders.person_recommendation.get_engine')
    @patch('data_ingestion.loaders.person_recommendation.get_central_engine')
    def test_run_success(self, mock_get_central_engine, mock_get_engine, mock_read_sql, mock_base_loader):
        """Test successful execution of person recommendation loader"""
        from data_ingestion.loaders import person_recommendation
        
        # Setup mocks
        mock_postgres_engine = Mock()
        mock_central_engine = Mock()
        mock_loader_instance = Mock()
        
        mock_get_engine.return_value = mock_postgres_engine
        mock_get_central_engine.return_value = mock_central_engine
        mock_base_loader.return_value = mock_loader_instance
        
        # Mock the SQL query result
        mock_df = pd.DataFrame({
            'id_recommendation': [1, 2],
            'id_person': [10, 20],
            'id_person_recommendation': [30, 40],
            'referral': [1, 0],
            'success_closed': [1, 0],
            'closed_date': ['2023-01-01 10:00:00', '0000-00-00 00:00:00']
        })
        mock_read_sql.return_value = mock_df
        
        # Execute
        person_recommendation.run()
        
        # Verify
        mock_get_engine.assert_called_once()
        mock_get_central_engine.assert_called_once()
        mock_read_sql.assert_called_once()
        mock_loader_instance.load.assert_called_once()
    
    @patch('data_ingestion.loaders.person_recommendation.get_engine')
    @patch('data_ingestion.loaders.person_recommendation.get_central_engine')
    def test_database_connection_failure(self, mock_get_central_engine, mock_get_engine):
        """Test handling of database connection failures"""
        from data_ingestion.loaders import person_recommendation
        
        # Setup
        mock_get_engine.side_effect = Exception("PostgreSQL connection failed")
        
        # Execute - should handle connection errors gracefully
        with pytest.raises(Exception, match="PostgreSQL connection failed"):
            person_recommendation.run()
    
    @patch('data_ingestion.loaders.person_recommendation.BaseLoader')
    @patch('data_ingestion.loaders.person_recommendation.pd.read_sql')
    @patch('data_ingestion.loaders.person_recommendation.get_engine')
    @patch('data_ingestion.loaders.person_recommendation.get_central_engine')
    def test_no_parameters_required(self, mock_get_central_engine, mock_get_engine, mock_read_sql, mock_base_loader):
        """Test that the loader doesn't require S3 parameters"""
        from data_ingestion.loaders import person_recommendation
        
        # Setup
        mock_get_engine.return_value = Mock()
        mock_get_central_engine.return_value = Mock()
        mock_base_loader.return_value = Mock()
        
        # Mock empty result with required columns
        mock_df = pd.DataFrame(columns=[
            'id_recommendation', 'id_person', 'id_person_recommendation',
            'referral', 'success_closed', 'closed_date'
        ])
        mock_read_sql.return_value = mock_df
        
        # Execute - should work without parameters
        person_recommendation.run()
        
        # Verify both engines are called
        mock_get_engine.assert_called_once()
        mock_get_central_engine.assert_called_once()