import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float

from data_ingestion.loaders.base_loader import BaseLoader
from data_ingestion.data_sources.config import Base

# Test model for BaseLoader testing  
class DummyModel(Base):
    __tablename__ = 'test_table'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    value = Column(Float, nullable=True)
    created_date = Column(DateTime, nullable=True)

class TestBaseLoader:
    
    def test_initialization(self):
        """Test BaseLoader initialization"""
        mock_engine = Mock()
        loader = BaseLoader(DummyModel, mock_engine)
        
        assert loader.model_class == DummyModel
        assert loader.engine == mock_engine
    
    @patch('data_ingestion.loaders.base_loader.inspect')
    def test_reset_table_new_table(self, mock_inspect):
        """Test _reset_table when table doesn't exist"""
        # Setup
        mock_engine = Mock()
        mock_inspector = Mock()
        mock_inspector.has_table.return_value = False
        mock_inspect.return_value = mock_inspector
        
        mock_table = Mock()
        mock_table.name = 'test_table'
        mock_table.schema = 'external'
        DummyModel.__table__ = mock_table
        
        loader = BaseLoader(DummyModel, mock_engine)
        
        # Execute
        loader._reset_table()
        
        # Verify
        mock_inspector.has_table.assert_called_once_with('test_table', schema='external')
        mock_table.drop.assert_not_called()  # Should not drop if doesn't exist
        mock_table.create.assert_called_once_with(mock_engine)
    
    @patch('data_ingestion.loaders.base_loader.inspect')
    def test_reset_table_existing_table(self, mock_inspect):
        """Test _reset_table when table already exists"""
        # Setup
        mock_engine = Mock()
        mock_inspector = Mock()
        mock_inspector.has_table.return_value = True
        mock_inspect.return_value = mock_inspector
        
        mock_table = Mock()
        mock_table.name = 'test_table'
        mock_table.schema = 'external'
        DummyModel.__table__ = mock_table
        
        loader = BaseLoader(DummyModel, mock_engine)
        
        # Execute
        loader._reset_table()
        
        # Verify
        mock_inspector.has_table.assert_called_once_with('test_table', schema='external')
        mock_table.drop.assert_called_once_with(mock_engine)  # Should drop existing table
        mock_table.create.assert_called_once_with(mock_engine)
    
    def test_normalize_records_basic(self):
        """Test _normalize_records with basic data types"""
        # Setup
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10.5, 20.0, None]
        })
        
        # Execute
        result = BaseLoader._normalize_records(df)
        
        # Verify structure and values
        assert len(result) == 3
        assert result[0] == {'id': 1, 'name': 'A', 'value': 10.5}
        assert result[1] == {'id': 2, 'name': 'B', 'value': 20.0}
        assert result[2]['id'] == 3
        assert result[2]['name'] == 'C'
        assert result[2]['value'] is None
    
    def test_normalize_records_with_datetime(self):
        """Test _normalize_records with datetime columns"""
        # Setup
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['A', 'B'],
            'created_date': [datetime(2023, 1, 1, 10, 0, 0), pd.NaT]
        })
        
        # Execute
        result = BaseLoader._normalize_records(df)
        
        # Verify
        assert len(result) == 2
        assert result[0]['created_date'] == datetime(2023, 1, 1, 10, 0, 0)
        assert result[1]['created_date'] is None
    
    def test_normalize_records_with_nan_values(self):
        """Test _normalize_records handles various NaN types"""
        # Setup
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', None, 'C'],
            'value': [10.5, float('nan'), pd.NA]
        })
        
        # Execute
        result = BaseLoader._normalize_records(df)
        
        # Verify - check structure and None conversion
        assert len(result) == 3
        assert result[0]['id'] == 1
        assert result[0]['name'] == 'A'
        assert result[0]['value'] == 10.5
        
        assert result[1]['id'] == 2
        assert result[1]['name'] is None
        assert result[1]['value'] is None  # float('nan') should become None
        
        assert result[2]['id'] == 3
        assert result[2]['name'] == 'C'
        assert result[2]['value'] is None  # pd.NA should become None
    
    @patch.object(BaseLoader, '_reset_table')
    def test_load_to_sql_method(self, mock_reset_table):
        """Test load() with to_sql method"""
        # Setup
        mock_engine = Mock()
        loader = BaseLoader(DummyModel, mock_engine)
        
        df = pd.DataFrame({
            'id': [1, 2],
            'name': ['A', 'B'],
            'value': [10.5, 20.0]
        })
        
        # Mock DataFrame.to_sql
        with patch.object(df, 'to_sql') as mock_to_sql:
            # Execute
            loader.load(df, method="to_sql", chunksize=500)
            
            # Verify
            mock_reset_table.assert_called_once()
            mock_to_sql.assert_called_once_with(
                name=DummyModel.__tablename__,
                con=mock_engine,
                schema=DummyModel.__table__.schema,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=500
            )
    
    @patch.object(BaseLoader, '_reset_table')
    @patch.object(BaseLoader, '_normalize_records')
    def test_load_bulk_core_method(self, mock_normalize, mock_reset_table):
        """Test load() with bulk_core method"""
        # Setup
        mock_engine = Mock()
        mock_connection = Mock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_connection
        mock_context_manager.__exit__.return_value = None
        mock_engine.begin.return_value = mock_context_manager
        
        loader = BaseLoader(DummyModel, mock_engine)
        
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10.5, 20.0, 30.0]
        })
        
        mock_normalize.return_value = [
            {'id': 1, 'name': 'A', 'value': 10.5},
            {'id': 2, 'name': 'B', 'value': 20.0},
            {'id': 3, 'name': 'C', 'value': 30.0}
        ]
        
        # Execute
        with patch('data_ingestion.loaders.base_loader.insert'):
            loader.load(df, method="bulk_core", chunksize=2)
            
            # Verify
            mock_reset_table.assert_called_once()
            assert mock_normalize.call_count == 2  # 2 chunks (2 + 1 rows)
            assert mock_connection.execute.call_count == 2  # 2 chunks
    
    @patch.object(BaseLoader, '_reset_table')
    def test_load_invalid_method(self, mock_reset_table):
        """Test load() with invalid method raises ValueError"""
        # Setup
        mock_engine = Mock()
        loader = BaseLoader(DummyModel, mock_engine)
        
        df = pd.DataFrame({'id': [1], 'name': ['A']})
        
        # Execute & Verify
        with pytest.raises(ValueError, match="Unknown method: invalid_method"):
            loader.load(df, method="invalid_method")
            
        # Verify _reset_table was still called before the error
        mock_reset_table.assert_called_once()
    
    @patch.object(BaseLoader, '_reset_table')
    def test_load_empty_dataframe(self, mock_reset_table):
        """Test load() with empty DataFrame"""
        # Setup
        mock_engine = Mock()
        loader = BaseLoader(DummyModel, mock_engine)
        
        df = pd.DataFrame(columns=['id', 'name', 'value'])
        
        # Execute - should not raise exception
        with patch.object(df, 'to_sql') as mock_to_sql:
            loader.load(df, method="to_sql")
            
            # Verify
            mock_reset_table.assert_called_once()
            mock_to_sql.assert_called_once()
    
    def test_normalize_records_preserves_dataframe(self):
        """Test that _normalize_records doesn't modify original DataFrame"""
        # Setup
        original_df = pd.DataFrame({
            'id': [1, 2],
            'name': ['A', 'B'],
            'value': [10.5, None]
        })
        original_copy = original_df.copy()
        
        # Execute
        BaseLoader._normalize_records(original_df)
        
        # Verify original DataFrame is unchanged
        pd.testing.assert_frame_equal(original_df, original_copy)
    
    @patch.object(BaseLoader, '_reset_table')
    def test_load_large_dataset_chunking(self, mock_reset_table):
        """Test load() properly handles chunking for large datasets"""
        # Setup
        mock_engine = Mock()
        mock_connection = Mock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_connection
        mock_context_manager.__exit__.return_value = None
        mock_engine.begin.return_value = mock_context_manager
        
        loader = BaseLoader(DummyModel, mock_engine)
        
        # Create larger dataset (5 rows, chunksize 2 = 3 chunks)
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['A', 'B', 'C', 'D', 'E'],
            'value': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        # Execute
        with patch('data_ingestion.loaders.base_loader.insert'):
            with patch.object(BaseLoader, '_normalize_records', return_value=[]) as mock_normalize:
                loader.load(df, method="bulk_core", chunksize=2)
                
                # Verify chunking occurred correctly
                mock_reset_table.assert_called_once()
                assert mock_normalize.call_count == 3  # 3 chunks: [0:2], [2:4], [4:5]
                assert mock_connection.execute.call_count == 3
    
    def test_normalize_records_with_mixed_datetime_formats(self):
        """Test _normalize_records handles different datetime formats"""
        # Setup
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'date_col': [
                datetime(2023, 1, 1),
                pd.Timestamp('2023-02-01'),
                pd.NaT
            ]
        })
        
        # Execute
        result = BaseLoader._normalize_records(df)
        
        # Verify
        assert len(result) == 3
        assert result[0]['date_col'] == datetime(2023, 1, 1)
        assert result[1]['date_col'] == pd.Timestamp('2023-02-01')
        assert result[2]['date_col'] is None