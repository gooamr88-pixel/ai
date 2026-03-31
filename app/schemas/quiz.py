"""
Ruya — Question Bank Schema (Flat JSON)
=========================================
Clean, flat schema for question bank generation.
Each question has: question text, 4 named options (a/b/c/d), correct answer letter.
"""

from pydantic import BaseModel, Field
from typing import List, Literal


class QuestionBankQuestion(BaseModel):
    """A single question with 4 flat options and a correct answer letter."""
    question: str = Field(..., description="The question text in Arabic")
    option_a: str = Field(..., description="Option A text")
    option_b: str = Field(..., description="Option B text")
    option_c: str = Field(..., description="Option C text")
    option_d: str = Field(..., description="Option D text")
    correct: Literal["a", "b", "c", "d"] = Field(..., description="The correct answer letter")


class QuestionBankResponse(BaseModel):
    """Full question bank response — flat list of questions."""
    questions: List[QuestionBankQuestion] = Field(..., description="List of 50 questions")
