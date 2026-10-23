"""Refinement-loop prompts: round 0 sends ``single_call``'s SuperCoder prompt; every later round
appends the ``methods/feedback.py`` feedback on the last candidate, so all feedback-driven methods
report a build/test/time result the same way."""

from __future__ import annotations

PROMPT_ID = "refine_feedback"
