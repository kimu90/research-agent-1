from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Union
from .common.model_schemas import ResearchToolOutput, AnalysisResult

class ResearchTool(BaseTool):
    """
    Base class for all research and analysis tools.
    Inherits from langchain.tools.BaseTool.
    """
    name: str
    description: str
    args_schema: Type[BaseModel]
    include_summary: bool = Field(default=False)

    def _run(self, **kwargs) -> Union[ResearchToolOutput, AnalysisResult]:
        """
        The main logic of the tool. Must be implemented by subclasses.

        Args:
            kwargs: Arbitrary keyword arguments specific to the tool's functionality.

        Returns:
            Union[ResearchToolOutput, AnalysisResult]: The result of the tool's execution.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def invoke(self, **kwargs) -> Union[ResearchToolOutput, AnalysisResult]:
        """
        Invokes the tool with proper input handling.

        Args:
            kwargs: Arbitrary keyword arguments specific to the tool's functionality.

        Returns:
            Union[ResearchToolOutput, AnalysisResult]: The result of the tool's execution.
        """
        if input in kwargs:
            kwargs = kwargs["input"]
        return super().invoke(input=kwargs)