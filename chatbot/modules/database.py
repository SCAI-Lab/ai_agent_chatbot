"""Database models and operations using SQLAlchemy."""
import logging
import sqlite3
from typing import List, Optional

from sqlalchemy import Column, Integer, Numeric, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import DB_PATH, DATABASE_URL

logger = logging.getLogger(__name__)


Base = declarative_base()


class User(Base):
    """User model with personality traits."""
    __tablename__ = "user"

    id = Column(Integer, primary_key=True, autoincrement=True)
    openness = Column(Numeric, nullable=True)
    conscientiousness = Column(Numeric, nullable=True)
    extraversion = Column(Numeric, nullable=True)
    agreeableness = Column(Numeric, nullable=True)
    neuroticism = Column(Numeric, nullable=True)
    name = Column(String, nullable=True)


def init_db():
    """Initialize database and create tables.

    Returns:
        SQLAlchemy Session factory.
    """
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)

    # Create memories table using raw SQL (for legacy compatibility)
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                chat_session_id TEXT,
                text TEXT,
                embedding TEXT,
                entity TEXT,
                current_summary TEXT
            )
        """)
        conn.commit()

    return sessionmaker(bind=engine)


def store_personality_traits(speaker_name: str, predictions: List[float], session) -> None:
    """Persist Big Five personality traits for a speaker.

    Args:
        speaker_name: Name of the speaker.
        predictions: List of 5 personality trait values [EXT, NEU, AGR, CON, OPN].
        session: SQLAlchemy session.
    """
    if not speaker_name:
        logger.warning("No speaker name provided. Skipping Big Five persistence.")
        return

    if len(predictions) != 5:
        logger.error("Unexpected personality predictions shape; expected 5 values.")
        return

    extraversion, neuroticism, agreeableness, conscientiousness, openness = predictions

    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT id FROM user WHERE name = ?", (speaker_name,))
            row = c.fetchone()

            if row:
                c.execute(
                    """
                    UPDATE user
                    SET openness = ?, conscientiousness = ?, extraversion = ?, agreeableness = ?, neuroticism = ?
                    WHERE name = ?
                    """,
                    (openness, conscientiousness, extraversion, agreeableness, neuroticism, speaker_name),
                )
            else:
                c.execute(
                    """
                    INSERT INTO user (name, openness, conscientiousness, extraversion, agreeableness, neuroticism)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (speaker_name, openness, conscientiousness, extraversion, agreeableness, neuroticism),
                )
            conn.commit()

    except Exception as e:
        logger.error(f"Failed to persist Big Five traits: {e}")


def fetch_user_data(speaker_name: str, session) -> Optional[dict]:
    """Fetch user data from database.

    Args:
        speaker_name: Name of the speaker.
        session: SQLAlchemy session.

    Returns:
        User preferences dict or None.
    """
    if not speaker_name:
        return None

    try:
        user = session.query(User).filter_by(name=speaker_name).first()
        if user and hasattr(user, 'preferences') and user.preferences:
            import json
            return json.loads(user.preferences)
    except Exception as e:
        logger.error(f"Failed to fetch user data: {e}")

    return None
