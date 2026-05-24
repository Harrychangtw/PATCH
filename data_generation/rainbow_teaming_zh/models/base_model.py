from typing import Optional, Dict, Any, List
import logging
from core.model_manager import ModelManager
from dataclasses import dataclass

import logging
from typing import Optional, Dict, Any, Literal, Dict

import torch
import numpy as np

import torch
from dataclasses import dataclass
from config.constants import BATCH_SIZE_PER_ITERATION

@dataclass
class ModelConfig:
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 2048
    repeat_penalty: float = 1.1
    n_threads: Optional[int] = None
    n_batch: Optional[int] = 512
    n_ctx: Optional[int] = 32768
    f16_kv: bool = True
    use_cuda: bool = True
    gram_limit: Optional[int] = None
    n_gpu_layers: int = -1
    use_mmap: bool = False
    use_mlock: bool = False
    verbose: bool = True
    embedding: bool = False
    offload_kqv: bool = False

ModelRole = Literal["target", "mutator", "judge"]

class BaseModel:
    def __init__(self, role: ModelRole, **kwargs):
        """Initialize base model with stricter memory management."""
        self.model_path = kwargs.pop('model_path')
        self.role = role
        self.logger = logging.getLogger(__name__)

        model_config_params = {
            k: v for k, v in kwargs.items()
            if k in ModelConfig.__annotations__
        }
        self.model_config = ModelConfig(**model_config_params)
        self.model_config_dict = model_config_params  

        self.model_manager = ModelManager()
        self.model = None

    def load_model(self):
        """Load model with optimized memory management."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            self.model = self.model_manager.load_model_for_role(
                role=self.role,
                model_path=self.model_path,
                config=self.model_config_dict
            )
            if not self.model:
                raise RuntimeError(f"Failed to load model for role {self.role}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def ensure_model_loaded(self):
        """Ensure model is loaded with proper memory management."""
        try:
            if not self.model or self.model_manager.current_role != self.role:
                self.load_model()
        except Exception as e:
            self.logger.error(f"Error ensuring model loaded: {str(e)}")
            self.model_manager.unload_current_model()
            raise

    def unload_model(self):
        """Unload the model using the ModelManager."""
        try:
            self.model_manager.unload_current_model()
        except Exception as e:
            self.logger.error(f"Error unloading model: {str(e)}")

    def generate_text(self, prompt: str, role: str = "user") -> str:
        """Generate text using the model with automatic loading and token streaming."""
        try:
            self.ensure_model_loaded()

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": role, "content": prompt}
            ]

            
            full_response = ""
            
            
            response_iter = self.model.create_chat_completion(
                messages=messages,
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                top_k=self.model_config.top_k,
                max_tokens=self.model_config.max_tokens,    
                repeat_penalty=self.model_config.repeat_penalty,
                stream=True
            )

            
            try:
                for response in response_iter:
                    if 'choices' in response and response['choices']:
                        delta = response['choices'][0]['delta']
                        if 'content' in delta:
                            content = delta['content']
                            print(content, end='', flush=True)
                            full_response += content
                            
                
                print()
                
                
                self.logger.debug(f"Raw response: {full_response}")
                
                return full_response.strip()
                
            except Exception as e:
                self.logger.error(f"Streaming failed: {str(e)}")
                return full_response.strip() if full_response else ""

        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}", exc_info=True)
            return ""
    def generate_batch(self, prompts: List[str], batch_size: int = BATCH_SIZE_PER_ITERATION) -> List[str]:
        """Optimized batch generation."""
        try:
            self.ensure_model_loaded()
            responses = []
            
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                for prompt in batch:
                    response = self.generate_text(prompt)
                    responses.append(response)
                
            return responses
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {str(e)}")
            return [""] * len(prompts)
        
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'model_manager'):
            self.model_manager.unload_current_model()