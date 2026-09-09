"""Abstract base class shared by all 10 pipeline steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from pipeline.config import PipelineConfig

T = TypeVar("T")


class PipelineStep(ABC, Generic[T]):
    """Common interface every step (Step 1 .. Step 10) implements.

    Each step receives the central :class:`PipelineConfig` (created by Step 1)
    and returns a step-specific result, keeping inter-step contracts explicit.
    """

    #: Human readable step identifier, e.g. ``"step01_environment"``.
    name: str = "base"

    @abstractmethod
    def run(self, config: PipelineConfig, *args: Any, **kwargs: Any) -> T:
        """Execute the step and return its result."""
        raise NotImplementedError
