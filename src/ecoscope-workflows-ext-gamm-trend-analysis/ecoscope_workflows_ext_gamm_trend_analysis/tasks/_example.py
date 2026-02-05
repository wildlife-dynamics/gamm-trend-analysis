from typing import Annotated

from ecoscope_workflows_core.decorators import task
from pydantic import Field


@task
def add_one_thousand(value: Annotated[float, Field(default=0, description="value to add")] = 0) -> float:
    return value + 1000
