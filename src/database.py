import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError
from contextlib import contextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

Base = declarative_base()

class RecognizedPlate(Base):
    __tablename__ = 'recognized_plates'

    id = Column(Integer, primary_key=True)
    timestamp = Column(String) # Store as ISO format string
    video_path = Column(String)
    frame_number = Column(Integer)
    car_id = Column(Integer)
    license_plate_text = Column(String)
    license_plate_text_score = Column(Float)
    license_plate_bbox = Column(Text) # Store bbox as a string (e.g., JSON or simple string representation)

    def __repr__(self):
        return (f"<RecognizedPlate(id={self.id}, timestamp='{self.timestamp}', "
                f"video_path='{self.video_path}', frame_number={self.frame_number}, "
                f"car_id={self.car_id}, license_plate_text='{self.license_plate_text}', "
                f"score={self.license_plate_text_score})>")

class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False) # In a real app, store hashed passwords!
    is_admin = Column(Boolean, default=False)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', is_admin={self.is_admin})>"

class DatabaseSession:
    _instance = None
    _engine = None
    _session_factory = None
    _db_file = 'database.db' # Default database file name

    def __new__(cls, db_file=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if db_file:
                cls._db_file = db_file
            cls.initialize()
        return cls._instance

    @classmethod
    def initialize(cls):
        """Initialize database engine and session factory"""
        if not cls._engine:
            try:
                # Ensure database directory exists
                db_dir = os.path.dirname(cls._db_file)
                if db_dir and not os.path.exists(db_dir):
                    os.makedirs(db_dir)
                
                # Build SQLite connection URL
                db_url = f"sqlite:///{cls._db_file}"

                cls._engine = create_engine(
                    db_url,
                    echo=False, # Set to True to see SQL statements
                    pool_pre_ping=True,
                )
                Base.metadata.create_all(cls._engine) # Create tables if they don't exist
                cls._session_factory = sessionmaker(bind=cls._engine)
                logger.info(f"SQLite database engine and session factory initialized for {cls._db_file}")
            except OperationalError as e:
                logger.error(f"Failed to initialize database due to operational error: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                raise

    @classmethod
    @contextmanager
    def session(cls):
        """Get database session"""
        if not cls._session_factory:
            cls.initialize()

        session = cls._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @classmethod
    def close(cls):
        """Close database engine - should only be called when shutting down the application"""
        if cls._engine:
            cls._engine.dispose()
            cls._engine = None
            cls._session_factory = None
            logger.info("Database engine closed")

# Initialize the database with a default file if not explicitly provided
DatabaseSession()
