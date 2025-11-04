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
    measurement_count = Column(Integer, default=0, nullable=False)


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
    """Persist Big Five personality traits for a speaker using cumulative averaging.

    Each new measurement is averaged with previous measurements to provide
    increasingly accurate personality trait estimates over time.

    Args:
        speaker_name: Name of the speaker.
        predictions: List of 5 personality trait values [EXT, NEU, AGR, CON, OPN].
        session: SQLAlchemy session (unused, kept for compatibility).
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

            # Check if user exists and get current values
            c.execute(
                """
                SELECT extraversion, neuroticism, agreeableness, conscientiousness, openness, measurement_count
                FROM user WHERE name = ?
                """,
                (speaker_name,)
            )
            row = c.fetchone()

            if row:
                # User exists - calculate cumulative average
                old_ext, old_neu, old_agr, old_con, old_opn, count = row

                # Handle NULL values (treat as 0.5 neutral if no previous data)
                old_ext = float(old_ext) if old_ext is not None else 0.5
                old_neu = float(old_neu) if old_neu is not None else 0.5
                old_agr = float(old_agr) if old_agr is not None else 0.5
                old_con = float(old_con) if old_con is not None else 0.5
                old_opn = float(old_opn) if old_opn is not None else 0.5
                count = int(count) if count is not None else 0

                # Calculate new averages: new_avg = (old_avg * count + new_value) / (count + 1)
                new_count = count + 1
                new_extraversion = (old_ext * count + extraversion) / new_count
                new_neuroticism = (old_neu * count + neuroticism) / new_count
                new_agreeableness = (old_agr * count + agreeableness) / new_count
                new_conscientiousness = (old_con * count + conscientiousness) / new_count
                new_openness = (old_opn * count + openness) / new_count

                logger.info(
                    f"Updating personality for {speaker_name} "
                    f"(measurement #{new_count}): "
                    f"EXT {old_ext:.3f}->{new_extraversion:.3f}, "
                    f"NEU {old_neu:.3f}->{new_neuroticism:.3f}, "
                    f"AGR {old_agr:.3f}->{new_agreeableness:.3f}, "
                    f"CON {old_con:.3f}->{new_conscientiousness:.3f}, "
                    f"OPN {old_opn:.3f}->{new_openness:.3f}"
                )

                # Update with averaged values
                c.execute(
                    """
                    UPDATE user
                    SET extraversion = ?, neuroticism = ?, agreeableness = ?,
                        conscientiousness = ?, openness = ?, measurement_count = ?
                    WHERE name = ?
                    """,
                    (new_extraversion, new_neuroticism, new_agreeableness,
                     new_conscientiousness, new_openness, new_count, speaker_name),
                )
            else:
                # Insert new user with first measurement (count = 1)
                logger.info(
                    f"Creating new user {speaker_name} with first measurement: "
                    f"EXT {extraversion:.3f}, NEU {neuroticism:.3f}, "
                    f"AGR {agreeableness:.3f}, CON {conscientiousness:.3f}, "
                    f"OPN {openness:.3f}"
                )

                c.execute(
                    """
                    INSERT INTO user (name, extraversion, neuroticism, agreeableness,
                                     conscientiousness, openness, measurement_count)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    """,
                    (speaker_name, extraversion, neuroticism, agreeableness,
                     conscientiousness, openness),
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
