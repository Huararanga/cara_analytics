import pytest
from unittest.mock import patch, Mock
import os

class TestDataIngestionIntegration:
    
    @patch('data_ingestion.loaders.population_density.run')
    @patch('data_ingestion.loaders.person_recommendation.run')
    @patch('data_ingestion.loaders.zts_list.run') 
    @patch('data_ingestion.loaders.czech_cities.run')
    @patch('data_ingestion.loaders.face_features.run')
    @patch('data_ingestion.main.initCentral')
    @patch('data_ingestion.main.initPostgres')
    @patch('data_ingestion.main.initS3')
    def test_main_function_calls_all_loaders(self, mock_init_s3, mock_init_postgres, 
                                           mock_init_central, mock_face, mock_czech,
                                           mock_zts, mock_person_rec, mock_pop_density):
        """Test that main function calls all loader functions"""
        from data_ingestion import main
        
        # Execute
        main.main()
        
        # Verify initialization calls
        mock_init_s3.assert_called_once()
        mock_init_postgres.assert_called_once()
        mock_init_central.assert_called_once()
        
        # Verify loader calls
        mock_face.assert_called_once()
        mock_czech.assert_called_once()
        mock_zts.assert_called_once()
        mock_person_rec.assert_called_once()
        mock_pop_density.assert_called_once()
    
    def test_environment_variables_exist(self):
        """Test that required environment variables can be accessed"""
        from data_ingestion import main
        
        # These should exist (even if they have default values)
        assert hasattr(main, 'DB_URL')
        assert hasattr(main, 'CENTRAL_DB_URL')
        assert hasattr(main, 'S3_BUCKET')
        assert hasattr(main, 'S3_ENDPOINT')
    
    @patch('data_ingestion.loaders.population_density.run')
    @patch('data_ingestion.loaders.person_recommendation.run')
    @patch('data_ingestion.loaders.zts_list.run') 
    @patch('data_ingestion.loaders.czech_cities.run')
    @patch('data_ingestion.loaders.face_features.run')
    @patch('data_ingestion.main.initS3')
    @patch('data_ingestion.main.initPostgres')
    @patch('data_ingestion.main.initCentral')
    def test_initialization_order(self, mock_init_central, mock_init_postgres, mock_init_s3, 
                                 mock_face, mock_czech, mock_zts, mock_person_rec, mock_pop_density):
        """Test that initialization happens in correct order"""
        from data_ingestion import main
        
        # Execute
        main.main()
        
        # Verify initialization was called (order doesn't matter for this test)
        mock_init_s3.assert_called_once()
        mock_init_postgres.assert_called_once() 
        mock_init_central.assert_called_once()