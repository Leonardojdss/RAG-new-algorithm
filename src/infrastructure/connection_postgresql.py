import os
from typing import Optional
from urllib.parse import quote_plus
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from dotenv import load_dotenv
import logging

load_dotenv()

Base = declarative_base()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Classe singleton para gerenciar a conexão com PostgreSQL usando SQLAlchemy
    """
    _instance: Optional['DatabaseConnection'] = None
    _engine: Optional[Engine] = None
    _session_factory: Optional[sessionmaker] = None

    def __new__(cls) -> 'DatabaseConnection':
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._engine is None:
            self._initialize_connection()

    def _initialize_connection(self) -> None:
        """
        Inicializa a conexão com o banco de dados PostgreSQL
        """
        try:
            db_host = os.getenv('DB_HOST')
            db_port = os.getenv('DB_PORT')
            db_name = os.getenv('DB_NAME')
            db_user = os.getenv('DB_USER')
            db_password = quote_plus(os.getenv('DB_PASSWORD'))

            database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=require"

            self._engine = create_engine(
                database_url,
                echo=os.getenv('DB_ECHO', 'False').lower() == 'true',  # Log SQL queries
                pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
                max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '20')),
                pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '30')),
                pool_recycle=int(os.getenv('DB_POOL_RECYCLE', '3600')),
                pool_pre_ping=True
            )

            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False
            )

        except Exception as e:
            logger.error(f"Erro ao conectar com PostgreSQL: {e}")
            raise

    def get_engine(self) -> Engine:
        """
        Retorna a engine do SQLAlchemy
        """
        if self._engine is None:
            raise RuntimeError("Conexão com banco não foi inicializada")
        return self._engine

    def get_session(self) -> Session:
        """
        Retorna uma nova sessão do SQLAlchemy
        """
        if self._session_factory is None:
            raise RuntimeError("Session factory não foi inicializada")
        return self._session_factory()

    def test_connection(self) -> bool:
        """
        Testa a conexão com o banco de dados
        """
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                logger.info("Teste de conexão concluida com sucesso")
                return True
        except Exception as e:
            logger.error(f"Erro no teste de conexão: {e}")
            return False

    def close_connection(self) -> None:
        """
        Fecha a conexão com o banco de dados
        """
        if self._engine:
            self._engine.dispose()
            logger.info("Conexão com banco fechada")


class DatabaseSession:
    """
    Context manager para gerenciar sessões do banco de dados
    """
    def __init__(self):
        self.db = DatabaseConnection()
        self.session: Optional[Session] = None

    def __enter__(self) -> Session:
        self.session = self.db.get_session()
        return self.session

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            try:
                if exc_type is not None:
                    self.session.rollback()
                    logger.error(f"Erro na sessão, fazendo rollback: {exc_val}")
                else:
                    self.session.commit()
            except Exception as e:
                self.session.rollback()
                logger.error(f"Erro ao fazer commit, fazendo rollback: {e}")
                raise
            finally:
                self.session.close()

def get_database_connection() -> DatabaseConnection:
    """
    Retorna uma instância da conexão com o banco
    """
    return DatabaseConnection()

def get_db_session() -> Session:
    """
    Retorna uma nova sessão do banco
    """
    return DatabaseConnection().get_session()

# test connection
# if __name__ == "__main__":
#     db_conn = get_database_connection()
#     test_connection = db_conn.test_connection()