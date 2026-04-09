"""Configuration tasks for forest analysis."""

from typing import Annotated
from ecoscope_workflows_core.decorators import task
from pydantic import Field


@task
def set_title_var(
    title: Annotated[str, Field(title="")],
) -> str:
    return title


@task
def set_tree_cover_threshold(
    threshold: Annotated[
        float,
        Field(
            default=60.0,
            description="Minimum tree cover percentage (0–100) to classify a pixel as forest.",
            ge=0.0,
            le=100.0,
        ),
    ] = 60.0,
) -> float:
    """Set the tree cover threshold used across all forest analysis steps."""
    return threshold
