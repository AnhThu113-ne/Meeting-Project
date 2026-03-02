"""
Database setup: SQLite + SQLAlchemy ORM
Tables: meetings, transcripts, meeting_minutes

Chong phinh database:
- Cascade delete: xoa meeting → xoa transcript + minutes tu dong
- Index tren cac cot thuong query (meeting_id, speaker)
- Khong luu raw_transcript (da co bang transcripts rieng)
- VACUUM sau khi xoa nhieu ban ghi
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey, Index, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# SQLite database file
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "meeting_data.db")
DATABASE_URL = f"sqlite:///{os.path.abspath(DB_PATH)}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    # Pool settings de tranh connection leak
    pool_pre_ping=True,
)

# Bat WAL mode cho SQLite → ghi nhanh hon, doc/ghi song song
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    duration = Column(Float, default=0.0)
    status = Column(String(50), default="pending")  # pending, processing, completed, error
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    transcripts = relationship("Transcript", back_populates="meeting", cascade="all, delete")
    minutes = relationship("MeetingMinutes", back_populates="meeting", cascade="all, delete", uselist=False)


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True)
    speaker = Column(String(100), default="Unknown", index=True)  # index → query theo speaker nhanh
    text = Column(Text, nullable=False)
    start_time = Column(Float, default=0.0)
    end_time = Column(Float, default=0.0)

    meeting = relationship("Meeting", back_populates="transcripts")

    # Composite index: query transcript theo meeting va sap xep theo time
    __table_args__ = (
        Index("idx_transcript_meeting_time", "meeting_id", "start_time"),
    )


class MeetingMinutes(Base):
    __tablename__ = "meeting_minutes"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), unique=True, nullable=False)
    summary = Column(Text, nullable=False)
    # Khong luu raw_transcript o day vi da co bang transcripts rieng → tranh luu trung
    created_at = Column(DateTime, default=datetime.utcnow)

    meeting = relationship("Meeting", back_populates="minutes")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at: {os.path.abspath(DB_PATH)}")


def vacuum_db():
    """Thu hoi dung luong sau khi xoa nhieu ban ghi."""
    with engine.connect() as conn:
        conn.execute("VACUUM")
    print("Database VACUUM completed")


def get_db_stats() -> dict:
    """Kiem tra kich thuoc va so luong ban ghi trong DB."""
    db_size = os.path.getsize(os.path.abspath(DB_PATH)) if os.path.exists(os.path.abspath(DB_PATH)) else 0
    db = SessionLocal()
    try:
        from sqlalchemy import text
        n_meetings = db.execute(text("SELECT COUNT(*) FROM meetings")).scalar()
        n_transcripts = db.execute(text("SELECT COUNT(*) FROM transcripts")).scalar()
        n_minutes = db.execute(text("SELECT COUNT(*) FROM meeting_minutes")).scalar()
    finally:
        db.close()
    return {
        "db_size_mb": round(db_size / 1024 / 1024, 2),
        "meetings": n_meetings,
        "transcripts": n_transcripts,
        "meeting_minutes": n_minutes,
    }
