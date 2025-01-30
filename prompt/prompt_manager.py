import yaml
import os
from datetime import datetime
from typing import Dict, Optional
from prompt.prompt import Prompt

class PromptManager:
    """
    Manages the storage, retrieval, and compilation of prompts for a single agent type.
    """
    
    def __init__(self, agent_type: str, config_path: str = "prompt_config"):
        """
        Initialize PromptManager with storage configuration.

        Args:
            agent_type: Type of agent (e.g., "general", "research")
            config_path: Directory path for storing prompt configurations
        """
        self.agent_type = agent_type
        self.config_path = config_path
        self.prompts_file = os.path.join(config_path, f"{agent_type}_prompts.yaml")
        self.backup_dir = os.path.join(config_path, "backup")
        self.init_directories()
        
        # Initialize prompts storage
        self.prompts: Dict[str, Prompt] = {}
        
        # Load existing prompts
        self._load_prompts()

    def init_directories(self) -> None:
        """Create necessary directories for storing prompts and backups"""
        os.makedirs(self.config_path, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        if not os.path.exists(self.prompts_file):
            self._init_yaml()

    def _init_yaml(self) -> None:
        """Initialize the YAML file with default structure"""
        initial_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0"
            },
            "prompts": {}
        }
        with open(self.prompts_file, 'w') as f:
            yaml.safe_dump(initial_data, f, sort_keys=False)

    def _create_backup(self) -> None:
        """Create a backup of the current prompts file"""
        if os.path.exists(self.prompts_file):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(self.backup_dir, f"{self.agent_type}_prompts_backup_{timestamp}.yaml")
            with open(self.prompts_file, 'r') as src, open(backup_file, 'w') as dst:
                dst.write(src.read())

    def _load_prompts(self) -> None:
        """Load prompts from YAML file into memory"""
        try:
            with open(self.prompts_file, 'r') as f:
                data = yaml.safe_load(f)
                for prompt_id, prompt_data in data["prompts"].items():
                    prompt = Prompt.from_dict(prompt_data)
                    self.prompts[prompt_id] = prompt
        except FileNotFoundError:
            pass  # File doesn't exist yet, no prompts to load
        except Exception as e:
            print(f"Error loading prompts: {str(e)}")

    def get_prompt(self, prompt_id: str) -> Optional[Prompt]:
        """
        Retrieve a prompt by ID.

        Args:
            prompt_id: Unique identifier of the prompt

        Returns:
            Optional[Prompt]: The prompt if found, None otherwise
        """
        return self.prompts.get(prompt_id)

    def add_prompt(self, prompt_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        """
        Add or update a prompt.

        Args:
            prompt_id: Unique identifier for the prompt
            content: The prompt template content
            metadata: Optional metadata for the prompt
        """
        # Create backup before modifying
        self._create_backup()
        
        # Create new prompt
        prompt = Prompt(
            id=prompt_id,
            content=content,
            metadata=metadata or {}
        )
        
        # Add to memory
        self.prompts[prompt_id] = prompt
        
        # Update YAML file
        with open(self.prompts_file, 'r') as f:
            data = yaml.safe_load(f)
        
        data["prompts"][prompt_id] = prompt.to_dict()
        
        with open(self.prompts_file, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def delete_prompt(self, prompt_id: str) -> bool:
        """
        Delete a prompt.

        Args:
            prompt_id: ID of the prompt to delete

        Returns:
            bool: True if prompt was deleted, False if not found
        """
        if prompt_id not in self.prompts:
            return False
            
        self._create_backup()
        
        # Remove from memory
        del self.prompts[prompt_id]
        
        # Remove from YAML
        with open(self.prompts_file, 'r') as f:
            data = yaml.safe_load(f)
        
        del data["prompts"][prompt_id]
            
        with open(self.prompts_file, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
            
        return True

    def compile_prompt(self, prompt_id: str, **kwargs) -> str:
        """
        Compile a prompt with provided variables.

        Args:
            prompt_id: ID of the prompt to compile
            **kwargs: Variables to use in compilation

        Returns:
            str: Compiled prompt with variables replaced

        Raises:
            ValueError: If prompt is not found
        """
        prompt = self.get_prompt(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt {prompt_id} not found")
        
        return prompt.compile(**kwargs)

    def list_prompts(self) -> Dict[str, Dict]:
        """
        List all available prompts with their metadata.

        Returns:
            Dict[str, Dict]: Dictionary of prompt IDs and their metadata
        """
        return {prompt_id: prompt.metadata for prompt_id, prompt in self.prompts.items()}