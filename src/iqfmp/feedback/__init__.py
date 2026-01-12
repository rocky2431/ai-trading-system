"""Feedback module for closed-loop factor mining.

This module implements the core feedback loop architecture for LLM-driven
factor discovery, based on the RL conceptual framework:
    State(hypothesis) -> Action(generate) -> Reward(evaluate) -> State'(refined)

Components:
- StructuredFeedback: Structured feedback data from evaluation results
- PatternMemory: Success/failure pattern storage and retrieval
- FeedbackPromptBuilder: LLM prompt construction with feedback context
- FeedbackLoop: Closed-loop coordinator for iterative factor mining

Academic validation:
- Chain-of-Alpha (arXiv 2508.06312)
- EvoAlpha (NeurIPS 2025)
- CogAlpha (arXiv 2511.18850)
"""

from iqfmp.feedback.feedback_loop import (
    FeedbackLoop,
    IterationResult,
    LoopConfig,
    LoopResult,
)
from iqfmp.feedback.pattern_memory import (
    PatternMemory,
    PatternRecord,
)
from iqfmp.feedback.prompt_builder import (
    FeedbackPromptBuilder,
)
from iqfmp.feedback.structured_feedback import (
    FailureReason,
    StructuredFeedback,
)

__all__ = [
    # Structured Feedback
    "FailureReason",
    "StructuredFeedback",
    # Pattern Memory
    "PatternMemory",
    "PatternRecord",
    # Prompt Builder
    "FeedbackPromptBuilder",
    # Feedback Loop
    "FeedbackLoop",
    "LoopConfig",
    "LoopResult",
    "IterationResult",
]
