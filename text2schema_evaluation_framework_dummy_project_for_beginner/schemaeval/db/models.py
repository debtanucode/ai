"""SQLAlchemy ORM models."""
from datetime import datetime
from sqlalchemy import JSON, DateTime, Float, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RunRecord(Base):
    __tablename__ = "run_records"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False)
    passed: Mapped[bool] = mapped_column(nullable=False)
    scores_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    tags_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    full_result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GoldenRecordRow(Base):
    __tablename__ = "golden_records"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="")
    schema_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    tags_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
