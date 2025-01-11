import logging
import os
import re

class Prompt:
    def __init__(self, prompt_id):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        prompt_folder = os.path.join(current_dir, "prompt_files")
        
        # Read all prompts.txt files. The file name is the prompt id.
        prompts = {}
        for file in os.listdir(prompt_folder):
            with open(os.path.join(prompt_folder, file), "r") as f:
                prompts[file.split(".")[0]] = f.read()
        
        self.template = prompts.get(prompt_id)
        
        if self.template is None:
            raise ValueError(
                f"Prompt with id {prompt_id} not found in folder {prompt_folder}"
            )

    def compile(self, **kwargs):
        def replace(match):
            var_name = match.group(1)
            return kwargs.get(var_name, match.group(0))
        
        # Use a custom function to replace only the double curly braces
        return re.sub(r"{{\s*(\w+)\s*}}", replace, self.template)