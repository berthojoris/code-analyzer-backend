"""
Database connection and session management.
"""
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from core.config import settings


class DatabaseManager:
    """Manager for database connections and sessions."""

    def __init__(self):
        self.engine: Optional[create_engine] = None
        self.SessionLocal: Optional[sessionmaker] = None

    def initialize(self):
        """Initialize database engine and session factory."""
        if self.engine is None:
            # Create engine with appropriate configuration for development/production
            if settings.database_url.startswith("sqlite"):
                # SQLite configuration for development
                self.engine = create_engine(
                    settings.database_url,
                    connect_args={
                        "check_same_thread": False,
                        "timeout": 20,
                    },
                    poolclass=StaticPool,
                    echo=settings.debug,
                )
            else:
                # PostgreSQL configuration for production
                self.engine = create_engine(
                    settings.database_url,
                    pool_size=settings.database_pool_size,
                    max_overflow=settings.database_max_overflow,
                    echo=settings.debug,
                )

            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )

    def create_tables(self):
        """Create all database tables."""
        if self.engine is None:
            self.initialize()

        from .models import Base
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session."""
        if self.engine is None:
            self.initialize()

        session = self.SessionLocal()  # type: ignore
        try:
            yield session
        finally:
            session.close()


# Global database manager instance
db_manager = DatabaseManager()


def get_db_session() -> Generator[Session, None, None]:
    """Dependency function to get database session."""
    yield from db_manager.get_session()


def initialize_database():
    """Initialize database and create tables."""
    db_manager.initialize()
    db_manager.create_tables()