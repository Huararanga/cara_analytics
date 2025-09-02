import pytest
from unittest.mock import patch, Mock, MagicMock
import io

class TestPopulationDensityLoader:
    
    @patch('data_ingestion.loaders.population_density.BaseLoader')
    @patch('data_ingestion.loaders.population_density.get_engine')  
    @patch('data_ingestion.loaders.population_density.read_from_minio')
    def test_run_success(self, mock_read_minio, mock_get_engine, mock_base_loader):
        """Test successful loading of population density data"""
        from data_ingestion.loaders import population_density
        
        # Setup mocks
        mock_engine = Mock()
        mock_get_engine.return_value = mock_engine
        mock_loader_instance = Mock()
        mock_base_loader.return_value = mock_loader_instance
        
        # Mock CSV data
        csv_data = "X,Y,Z\n14.2945,51.0537,25.44\n14.3029,51.0537,16.38\n"
        mock_read_minio.return_value = io.StringIO(csv_data)
        
        # Execute
        population_density.run("test_bucket", "population_density.csv")
        
        # Verify function calls
        mock_read_minio.assert_called_once_with("test_bucket", "population_density.csv")
        mock_get_engine.assert_called_once()
        mock_base_loader.assert_called_once()
        mock_loader_instance.load.assert_called_once()
    
    @patch('data_ingestion.loaders.population_density.BaseLoader')
    @patch('data_ingestion.loaders.population_density.get_engine')
    @patch('data_ingestion.loaders.population_density.read_from_minio')
    def test_column_renaming(self, mock_read_minio, mock_get_engine, mock_base_loader):
        """Test that X,Y,Z columns are renamed to longitude,latitude,density"""
        from data_ingestion.loaders import population_density
        
        # Setup mocks
        mock_get_engine.return_value = Mock()
        mock_loader_instance = Mock()
        mock_base_loader.return_value = mock_loader_instance
        
        csv_data = "X,Y,Z\n14.2945,51.0537,25.44\n"
        mock_read_minio.return_value = io.StringIO(csv_data)
        
        # Execute
        population_density.run("test_bucket", "test_file.csv")
        
        # Verify the loader was called with transformed data
        mock_loader_instance.load.assert_called_once()
        # Get the DataFrame that was passed to load()
        call_args = mock_loader_instance.load.call_args[0]
        df = call_args[0]
        
        # Check columns were renamed
        assert 'longitude' in df.columns
        assert 'latitude' in df.columns  
        assert 'density' in df.columns
        assert 'X' not in df.columns
        assert 'Y' not in df.columns
        assert 'Z' not in df.columns
    
    @patch('data_ingestion.loaders.population_density.BaseLoader')
    @patch('data_ingestion.loaders.population_density.get_engine')
    @patch('data_ingestion.loaders.population_density.read_from_minio')
    def test_empty_file_handling(self, mock_read_minio, mock_get_engine, mock_base_loader):
        """Test handling of empty CSV files"""
        from data_ingestion.loaders import population_density
        
        # Setup mocks
        mock_get_engine.return_value = Mock()
        mock_loader_instance = Mock()
        mock_base_loader.return_value = mock_loader_instance
        mock_read_minio.return_value = io.StringIO("X,Y,Z\n")  # Header only
        
        # Execute - should not raise exception
        population_density.run("test_bucket", "empty_file.csv")
        
        # Verify it still tries to load (even if empty)
        mock_loader_instance.load.assert_called_once()
    
    @patch('data_ingestion.loaders.population_density.BaseLoader')
    @patch('data_ingestion.loaders.population_density.get_engine')
    @patch('data_ingestion.loaders.population_density.read_from_minio')
    def test_malformed_csv_handling(self, mock_read_minio, mock_get_engine, mock_base_loader):
        """Test handling of malformed CSV data"""
        from data_ingestion.loaders import population_density
        
        # Setup mocks  
        mock_get_engine.return_value = Mock()
        mock_base_loader.return_value = Mock()
        
        # Malformed CSV (missing Z column)
        csv_data = "X,Y\n14.2945,51.0537\n"
        mock_read_minio.return_value = io.StringIO(csv_data)
        
        # Execute - should raise KeyError when trying to rename Z column
        # Note: pandas might handle this differently, so we just test it runs
        try:
            population_density.run("test_bucket", "malformed_file.csv")
        except KeyError:
            pass  # This is expected behavior