from datetime import datetime, timedelta
import uuid
from enum import Enum as PyEnum
from pydantic import BaseModel
from typing import List, Optional

# Session statuses
class SessionStatus(str, PyEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    EXPIRED = "expired"
    TERMINATED = "terminated"

# Pydantic model for data validation
class InterviewSession(BaseModel):
    id: str = str(uuid.uuid4())
    user_id: str
    start_time: datetime = datetime.utcnow()
    expiry_time: datetime = datetime.utcnow() + timedelta(minutes=30)
    status: SessionStatus = SessionStatus.ACTIVE

class InterviewRequestCreate(BaseModel):
    job_title: str
    experience_years: int
    seniority_level: str
    skills: Optional[List[str]] = None
    industry: Optional[str] = None
    num_questions: int = 5
    round_type: Optional[str] = None
    user_instructions: Optional[str] = None
    user_id: str  # uuid from users table