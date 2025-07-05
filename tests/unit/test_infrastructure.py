import pytest
from unittest.mock import patch, MagicMock, Mock
from sqlalchemy.exc import SQLAlchemyError
import os
import logging

from src.infrastructure.connection_postgresql import (
    DatabaseConnection,
    DatabaseSession,
    get_database_connection,
    get_db_session
)


class TestDatabaseConnection:
    """Test cases for DatabaseConnection class"""
    
    @patch.dict(os.environ, {
        'DB_HOST': 'localhost',
        'DB_PORT': '5432',
        'DB_NAME': 'test_db',
        'DB_USER': 'test_user',
        'DB_PASSWORD': 'test_password',
        'DB_ECHO': 'true',
        'DB_POOL_SIZE': '5',
        'DB_MAX_OVERFLOW': '10',
        'DB_POOL_TIMEOUT': '20',
        'DB_POOL_RECYCLE': '1800'
    })
    @patch('src.infrastructure.connection_postgresql.create_engine')
    @patch('src.infrastructure.connection_postgresql.sessionmaker')
    def test_initialize_connection_success(self, mock_sessionmaker, mock_create_engine):
        """Test successful database connection initialization"""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        mock_session_factory = Mock()
        mock_sessionmaker.return_value = mock_session_factory
        
        # Reset singleton instance
        DatabaseConnection._instance = None
        DatabaseConnection._engine = None
        DatabaseConnection._session_factory = None
        
        db_conn = DatabaseConnection()
        
        mock_create_engine.assert_called_once()
        mock_sessionmaker.assert_called_once()
        assert db_conn._engine == mock_engine
        assert db_conn._session_factory == mock_session_factory

    @patch.dict(os.environ, {
        'DB_HOST': 'localhost',
        'DB_PORT': '5432',
        'DB_NAME': 'test_db',
        'DB_USER': 'test_user',
        'DB_PASSWORD': 'test_password'
    })
    @patch('src.infrastructure.connection_postgresql.create_engine')
    def test_initialize_connection_failure(self, mock_create_engine):
        """Test database connection initialization failure"""
        mock_create_engine.side_effect = Exception("Connection failed")
        
        # Reset singleton instance
        DatabaseConnection._instance = None
        DatabaseConnection._engine = None
        DatabaseConnection._session_factory = None
        
        with pytest.raises(Exception, match="Connection failed"):
            DatabaseConnection()

    def test_singleton_pattern(self):
        """Test that DatabaseConnection follows singleton pattern"""
        # Reset singleton instance
        DatabaseConnection._instance = None
        DatabaseConnection._engine = None
        DatabaseConnection._session_factory = None
        
        with patch('src.infrastructure.connection_postgresql.create_engine'):
            with patch('src.infrastructure.connection_postgresql.sessionmaker'):
                db1 = DatabaseConnection()
                db2 = DatabaseConnection()
                
                assert db1 is db2

    @patch('src.infrastructure.connection_postgresql.create_engine')
    @patch('src.infrastructure.connection_postgresql.sessionmaker')
    def test_get_engine_success(self, mock_sessionmaker, mock_create_engine):
        """Test getting engine successfully"""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Reset singleton instance
        DatabaseConnection._instance = None
        DatabaseConnection._engine = None
        DatabaseConnection._session_factory = None
        
        db_conn = DatabaseConnection()
        engine = db_conn.get_engine()
        
        assert engine == mock_engine

    def test_get_engine_not_initialized(self):
        """Test getting engine when not initialized"""
        db_conn = DatabaseConnection.__new__(DatabaseConnection)
        db_conn._engine = None
        
        with pytest.raises(RuntimeError, match="Conexão com banco não foi inicializada"):
            db_conn.get_engine()

    @patch('src.infrastructure.connection_postgresql.create_engine')
    @patch('src.infrastructure.connection_postgresql.sessionmaker')
    def test_get_session_success(self, mock_sessionmaker, mock_create_engine):
        """Test getting session successfully"""
        mock_session_factory = Mock()
        mock_session = Mock()
        mock_session_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_factory
        
        # Reset singleton instance
        DatabaseConnection._instance = None
        DatabaseConnection._engine = None
        DatabaseConnection._session_factory = None
        
        db_conn = DatabaseConnection()
        session = db_conn.get_session()
        
        assert session == mock_session

    def test_get_session_not_initialized(self):
        """Test getting session when not initialized"""
        db_conn = DatabaseConnection.__new__(DatabaseConnection)
        db_conn._session_factory = None
        
        with pytest.raises(RuntimeError, match="Session factory não foi inicializada"):
            db_conn.get_session()

    @patch('src.infrastructure.connection_postgresql.create_engine')
    @patch('src.infrastructure.connection_postgresql.sessionmaker')
    def test_test_connection_success(self, mock_sessionmaker, mock_create_engine):
        """Test successful connection test"""
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        mock_session_factory = Mock()
        mock_session_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_factory
        
        # Reset singleton instance
        DatabaseConnection._instance = None
        DatabaseConnection._engine = None
        DatabaseConnection._session_factory = None
        
        db_conn = DatabaseConnection()
        
        result = db_conn.test_connection()
        
        assert result is True
        mock_session.execute.assert_called_once()

    @patch('src.infrastructure.connection_postgresql.create_engine')
    @patch('src.infrastructure.connection_postgresql.sessionmaker')
    def test_test_connection_failure(self, mock_sessionmaker, mock_create_engine):
        """Test failed connection test"""
        mock_session = Mock()
        mock_session.execute.side_effect = SQLAlchemyError("Connection test failed")
        mock_session_factory = Mock()
        mock_session_factory.return_value = mock_session
        mock_sessionmaker.return_value = mock_session_factory
        
        # Reset singleton instance
        DatabaseConnection._instance = None
        DatabaseConnection._engine = None
        DatabaseConnection._session_factory = None
        
        db_conn = DatabaseConnection()
        
        with patch.object(db_conn, 'get_session', return_value=mock_session):
            result = db_conn.test_connection()
            
            assert result is False

    @patch('src.infrastructure.connection_postgresql.create_engine')
    @patch('src.infrastructure.connection_postgresql.sessionmaker')
    def test_close_connection(self, mock_sessionmaker, mock_create_engine):
        """Test closing connection"""
        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine
        
        # Reset singleton instance
        DatabaseConnection._instance = None
        DatabaseConnection._engine = None
        DatabaseConnection._session_factory = None
        
        db_conn = DatabaseConnection()
        db_conn.close_connection()
        
        mock_engine.dispose.assert_called_once()


class TestDatabaseSession:
    """Test cases for DatabaseSession context manager"""
    
    @patch('src.infrastructure.connection_postgresql.DatabaseConnection')
    def test_context_manager_success(self, mock_db_connection_class):
        """Test successful context manager usage"""
        mock_db_instance = Mock()
        mock_session = Mock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_connection_class.return_value = mock_db_instance
        
        with DatabaseSession() as session:
            assert session == mock_session
        
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @patch('src.infrastructure.connection_postgresql.DatabaseConnection')
    def test_context_manager_with_exception(self, mock_db_connection_class):
        """Test context manager with exception"""
        mock_db_instance = Mock()
        mock_session = Mock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_connection_class.return_value = mock_db_instance
        
        try:
            with DatabaseSession() as session:
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        mock_session.rollback.assert_called_once()
        mock_session.close.assert_called_once()

    @patch('src.infrastructure.connection_postgresql.DatabaseConnection')
    def test_context_manager_commit_exception(self, mock_db_connection_class):
        """Test context manager when commit raises exception"""
        mock_db_instance = Mock()
        mock_session = Mock()
        mock_session.commit.side_effect = SQLAlchemyError("Commit failed")
        mock_db_instance.get_session.return_value = mock_session
        mock_db_connection_class.return_value = mock_db_instance
        
        with pytest.raises(SQLAlchemyError):
            with DatabaseSession() as session:
                pass
        
        mock_session.rollback.assert_called()
        mock_session.close.assert_called_once()


class TestUtilityFunctions:
    """Test cases for utility functions"""
    
    @patch('src.infrastructure.connection_postgresql.DatabaseConnection')
    def test_get_database_connection(self, mock_db_connection_class):
        """Test get_database_connection function"""
        mock_db_instance = Mock()
        mock_db_connection_class.return_value = mock_db_instance
        
        result = get_database_connection()
        
        assert result == mock_db_instance
        mock_db_connection_class.assert_called_once()

    @patch('src.infrastructure.connection_postgresql.DatabaseConnection')
    def test_get_db_session(self, mock_db_connection_class):
        """Test get_db_session function"""
        mock_db_instance = Mock()
        mock_session = Mock()
        mock_db_instance.get_session.return_value = mock_session
        mock_db_connection_class.return_value = mock_db_instance
        
        result = get_db_session()
        
        assert result == mock_session
        mock_db_instance.get_session.assert_called_once()
