import yaml
import os
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
from research_agent.tracers import QueryTrace
from .prompt import Prompt  # Import from local prompt.py

class AgentType(str, Enum):
    """Available agent types"""
    GENERAL = "general"



class PromptManager:
    """
    Manages the storage, retrieval, and compilation of prompts for different agent types.
    """
    
    def __init__(self, config_path: str = "prompt_config"):
        """
        Initialize PromptManager with storage configuration.

        Args:
            config_path: Directory path for storing prompt configurations
        """
        self.config_path = config_path
        self.prompts_file = os.path.join(config_path, "prompts.yaml")
        self.backup_dir = os.path.join(config_path, "backup")
        self.init_directories()
        
        # Initialize tracer with default query
        self.tracer = QueryTrace(query="initialization")
        
        # Initialize prompts storage
        self.prompts: Dict[str, Dict[str, Prompt]] = {
            agent_type.value: {} for agent_type in AgentType
        }
        
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
            "agents": {
                "general": {"prompts": {}},
            }
        }
        with open(self.prompts_file, 'w') as f:
            yaml.safe_dump(initial_data, f, sort_keys=False)

    def _create_backup(self) -> None:
        """Create a backup of the current prompts file"""
        if os.path.exists(self.prompts_file):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(self.backup_dir, f"prompts_backup_{timestamp}.yaml")
            with open(self.prompts_file, 'r') as src, open(backup_file, 'w') as dst:
                dst.write(src.read())

    def _load_prompts(self) -> None:
        """Load prompts from YAML file into memory"""
        try:
            with open(self.prompts_file, 'r') as f:
                data = yaml.safe_load(f)
                for agent_type in AgentType:
                    agent_prompts = data["agents"][agent_type.value]["prompts"]
                    for prompt_id, prompt_data in agent_prompts.items():
                        prompt = Prompt(
                            id=prompt_id,
                            content=prompt_data["content"],
                            metadata=prompt_data.get("metadata", {}),
                            created_at=datetime.fromisoformat(prompt_data["created_at"])
                        )
                        self.prompts[agent_type.value][prompt_id] = prompt
        except Exception as e:
            logging.error(f"Error loading prompts: {str(e)}")

    def get_prompt(self, prompt_id: str, agent_type: AgentType) -> Optional[Prompt]:
        """
        Retrieve a prompt by ID for a specific agent type.

        Args:
            prompt_id: Unique identifier of the prompt
            agent_type: Type of agent the prompt belongs to

        Returns:
            Optional[Prompt]: The prompt if found, None otherwise
        """
        return self.prompts[agent_type.value].get(prompt_id)

    def add_prompt(self, prompt_id: str, content: str, agent_type: AgentType, 
                  metadata: Optional[Dict] = None) -> None:
        """
        Add or update a prompt for a specific agent type.

        Args:
            prompt_id: Unique identifier for the prompt
            content: The prompt template content
            agent_type: Type of agent this prompt belongs to
            metadata: Optional metadata for the prompt
        """
        # Create backup before modifying
        self._create_backup()
        
        # Create new prompt
        prompt = Prompt(
            id=prompt_id,
            content=content,
            metadata=metadata or {},
        )
        
        # Add to memory
        self.prompts[agent_type.value][prompt_id] = prompt
        
        # Update YAML file
        with open(self.prompts_file, 'r') as f:
            data = yaml.safe_load(f)
        
        data["agents"][agent_type.value]["prompts"][prompt_id] = prompt.to_dict()
        
        with open(self.prompts_file, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)

    def delete_prompt(self, prompt_id: str, agent_type: AgentType) -> bool:
        """
        Delete a prompt.

        Args:
            prompt_id: ID of the prompt to delete
            agent_type: Type of agent the prompt belongs to

        Returns:
            bool: True if prompt was deleted, False if not found
        """
        if prompt_id not in self.prompts[agent_type.value]:
            return False
            
        self._create_backup()
        
        # Remove from memory
        del self.prompts[agent_type.value][prompt_id]
        
        # Remove from YAML
        with open(self.prompts_file, 'r') as f:
            data = yaml.safe_load(f)
        
        if prompt_id in data["agents"][agent_type.value]["prompts"]:
            del data["agents"][agent_type.value]["prompts"][prompt_id]
            
        with open(self.prompts_file, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
            
        return True

    def compile_prompt(self, prompt_id: str, agent_type: AgentType, **kwargs) -> str:
        """
        Compile a prompt with provided variables.

        Args:
            prompt_id: ID of the prompt to compile
            agent_type: Type of agent
            **kwargs: Variables to use in compilation

        Returns:
            str: Compiled prompt with variables replaced

        Raises:
            ValueError: If prompt is not found
        """
        prompt = self.get_prompt(prompt_id, agent_type)
        if not prompt:
            raise ValueError(f"Prompt {prompt_id} not found for agent {agent_type}")
        
        # Update tracer for this compilation
        self.tracer = QueryTrace(query=f"{prompt_id}_{agent_type}")
        trace_id = str(uuid.uuid4())
        
        self.tracer.log_step(trace_id, "compile_prompt", {
            "agent_type": agent_type,
            "prompt_id": prompt_id,
            "variables": kwargs
        })
        
        try:
            return prompt.compile(**kwargs)
        except Exception as e:
            self.tracer.log_step(trace_id, "compile_error", {
                "error": str(e)
            })
            raise

    def list_prompts(self, agent_type: Optional[AgentType] = None) -> Dict[str, List[str]]:
        """
        List all available prompts, optionally filtered by agent type.

        Args:
            agent_type: Optional filter for specific agent type

        Returns:
            Dict[str, List[str]]: Dictionary of agent types and their prompt IDs
        """
        if agent_type:
            return {agent_type.value: list(self.prompts[agent_type.value].keys())}
        return {agent_type: list(prompts.keys()) 
                for agent_type, prompts in self.prompts.items()}