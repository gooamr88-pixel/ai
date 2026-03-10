from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


# ── Request ──────────────────────────────────────────────────────────────────

class QuizRequest(BaseModel):
    """Request body for quiz generation."""
    text: str = Field(..., min_length=3, max_length=50000, description="Educational text to generate quiz from")
    num_questions: int = Field(default=5, ge=1, le=30, description="Number of questions to generate")
    difficulty: Difficulty = Field(default=Difficulty.medium, description="Desired difficulty level")


# ── Response ─────────────────────────────────────────────────────────────────

class QuizOption(BaseModel):
    """A single answer option."""
    text: str
    is_correct: bool


class QuizQuestion(BaseModel):
    """A single quiz question with options and explanation."""
    question: str
    options: List[QuizOption]
    explanation: str
    difficulty: str


class QuizResponse(BaseModel):
    """Full quiz response returned to the client."""
    id: Optional[str] = None
    title: str
    questions: List[QuizQuestion]


class QuestionBank(BaseModel):
    """Strictly English, Exactly 50 questions (30 MCQ, 20 T/F)."""
    id: Optional[str] = None
    title: str
    questions: List[QuizQuestion] = Field(..., description="Exactly 50 questions (30 MCQ, 20 True/False) in strictly English.", min_items=50, max_items=50)
