from datetime import datetime
import os
from typing import Dict, Optional
from pydantic import BaseModel

class Prompt(BaseModel):
    """
    Base class for prompts that handles prompt content and metadata
    """
    id: str
    content: str
    metadata: Optional[Dict] = {}
    created_at: datetime = datetime.now()

    class Config:
        """Allow arbitrary types for metadata"""
        arbitrary_types_allowed = True

    def compile(self, **kwargs) -> str:
        """
        Compile the prompt template with provided variables.
        
        Args:
            **kwargs: Dictionary of variables to replace in the template
            
        Returns:
            str: Compiled prompt with variables replaced
            
        Example:
            prompt = Prompt(
                id="research",
                content="Research {{topic}} focusing on {{aspect}}"
            )
            result = prompt.compile(topic="AI Safety", aspect="Technical Alignment")
        """
        compiled_content = self.content
        for key, value in kwargs.items():
            placeholder = f"{{{{{key}}}}}"
            compiled_content = compiled_content.replace(placeholder, str(value))
        return compiled_content

    @classmethod
    def from_dict(cls, data: Dict) -> "Prompt":
        """
        Create a Prompt instance from a dictionary.
        
        Args:
            data: Dictionary containing prompt data
            
        Returns:
            Prompt: New Prompt instance
        """
        return cls(
            id=data.get("id"),
            content=data.get("content"),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        )

    def to_dict(self) -> Dict:
        """
        Convert the prompt to a dictionary format.
        
        Returns:
            Dict: Dictionary representation of the prompt
        """
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

    def update_content(self, new_content: str) -> None:
        """
        Update the prompt's content.
        
        Args:
            new_content: New content for the prompt
        """
        self.content = new_content

    def update_metadata(self, new_metadata: Dict) -> None:
        """
        Update or add new metadata fields.
        
        Args:
            new_metadata: Dictionary of metadata to update/add
        """
        self.metadata.update(new_metadata)