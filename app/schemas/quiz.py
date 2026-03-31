"""
Ruya — Question Bank Schema
==============================
Nested format: each question has text, type (MCQ/TF), and options array.
"""

from pydantic import BaseModel, Field
from typing import List, Literal


class QuestionOption(BaseModel):
    """A single answer option."""
    text: str = Field(..., description="Option text")
    isCorrect: bool = Field(..., description="Whether this option is the correct answer")


class QuestionBankQuestion(BaseModel):
    """A single question with type and options."""
    text: str = Field(..., description="The question text")
    type: Literal["MCQ", "TF"] = Field(..., description="Question type: MCQ or TF")
    options: List[QuestionOption] = Field(..., description="List of 4 answer options")


class QuestionBankResponse(BaseModel):
    """Full question bank response."""
    questions: List[QuestionBankQuestion] = Field(..., description="List of 50 questions")
