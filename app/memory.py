"""Conversation memory storage backed by SQLite.

This module defines a simple persistence layer for storing and retrieving
messages exchanged with the LLM.  Messages are associated with a
``session_id`` so that each conversation can be isolated.  The
implementation uses SQLAlchemy for portability and ease of use.

For single‑process deployments, SQLite is sufficient.  When scaling the
service across multiple processes or machines, switch to a central
database by setting the ``DATABASE_URL`` environment variable.

Functions
---------
add_message(session_id: str, role: str, content: str, tokens: int) -> None
    Persist a new message.

get_messages(session_id: str) -> list[dict]
    Retrieve messages for a session in insertion order.

get_context(session_id: str) -> str
    Concatenate message contents to form the conversation context.

clear_session(session_id: str) -> None
    Delete all messages for a session.
"""

from typing import List
from sqlalchemy import Column, Integer, String, Text, create_engine, select, delete
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import get_settings

Base = declarative_base()


class Message(Base):
    """ORM model representing a single chat message."""

    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)  # "user" or "assistant"
    content = Column(Text, nullable=False)
    tokens = Column(Integer, nullable=False)


# Create the SQLAlchemy engine and session factory once at import time
settings = get_settings()
engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Create tables on first import
Base.metadata.create_all(bind=engine)


def add_message(session_id: str, role: str, content: str, tokens: int) -> None:
    """Persist a new chat message to the database."""
    db = SessionLocal()
    try:
        msg = Message(session_id=session_id, role=role, content=content, tokens=tokens)
        db.add(msg)
        db.commit()
    finally:
        db.close()


def get_messages(session_id: str) -> List[dict]:
    """Retrieve all messages for a session ordered by insertion."""
    db = SessionLocal()
    try:
        stmt = select(Message).where(Message.session_id == session_id).order_by(Message.id)
        rows = db.execute(stmt).scalars().all()
        return [
            {"role": row.role, "content": row.content, "tokens": row.tokens}
            for row in rows
        ]
    finally:
        db.close()


def get_context(session_id: str) -> str:
    """Concatenate message contents into a single context string."""
    messages = get_messages(session_id)
    return "\n".join(msg["content"] for msg in messages)


def clear_session(session_id: str) -> None:
    """Remove all messages belonging to a session."""
    db = SessionLocal()
    try:
        stmt = delete(Message).where(Message.session_id == session_id)
        db.execute(stmt)
        db.commit()
    finally:
        db.close()