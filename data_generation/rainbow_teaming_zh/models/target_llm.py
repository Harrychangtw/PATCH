
import os
import logging
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any
from .base_model import BaseModel
from core.model_manager import ModelManager
from utils.db_initializer import DatabaseInitializer
from config.constants import MODEL_CONFIG
DATA_DIR = Path("data")

class TargetLLM(BaseModel):
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)
        
        if 'model_path' not in kwargs:
            raise ValueError("model_path must be provided")
        
        super().__init__(role="target", **kwargs)
        self.db_path = self._get_model_db_path()

        
        if not os.path.exists(self.db_path):
            self._initialize_model_db()

    def _get_model_db_path(self) -> str:
        model_name = os.path.basename(self.model_path)
        db_file = DATA_DIR / f"{model_name}.db"
        return str(db_file)
    
    def _initialize_model_db(self):
        try:
            self.model_manager.unload_current_model()  
            initializer = DatabaseInitializer(db_path=self.db_path)
            initializer.init_database()
            initializer.init_with_seeds()
            initializer.cleanup()
        except Exception as e:
            self.logger.error(f"Database initialization failed: {str(e)}")
            raise

    def generate_batch(self, prompts: List[str]) -> List[str]:
        """Optimized batch generation with streaming."""
        results = []
        try:
            self.ensure_model_loaded()

            for idx, prompt in enumerate(prompts, 1):
                print(f"\nGenerating response for prompt {idx}/{len(prompts)}:")
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]

                
                full_response = ""
                
                
                response_iter = self.model.create_chat_completion(
                    messages=messages,
                    temperature=self.model_config.temperature,
                    top_p=self.model_config.top_p,
                    top_k=self.model_config.top_k,
                    repeat_penalty=self.model_config.repeat_penalty,
                    max_tokens=self.model_config.max_tokens,
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
                    
                    if full_response:
                        self.logger.debug(f"Generated content: {full_response[:30]}...")
                        results.append(full_response.strip())
                    else:
                        self.logger.error(f"No response received for prompt: {prompt}")
                        results.append("")

                except Exception as e:
                    self.logger.error(f"Streaming failed for prompt {idx}: {str(e)}")
                    results.append("")

        except Exception as e:
            self.logger.error(f"Batch generation failed: {str(e)}", exc_info=True)
            results.extend([""] * (len(prompts) - len(results)))

        return results
    
    def generate_response(self, prompt: str) -> str:
        """Generate single response with streaming."""
        try:
            self.ensure_model_loaded()
            
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]

            
            full_response = ""
            
            
            response_iter = self.model.create_chat_completion(
                messages=messages,
                temperature=self.model_config.temperature,
                top_p=self.model_config.top_p,
                top_k=self.model_config.top_k,
                repeat_penalty=self.model_config.repeat_penalty,
                max_tokens=self.model_config.max_tokens,
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
                return full_response.strip()

            except Exception as e:
                self.logger.error(f"Streaming failed: {str(e)}")
                return full_response.strip() if full_response else ""

        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}")
            return ""

    def cleanup(self):
        """Enhanced cleanup."""
        try:
            if self.model_manager.current_role == "target":
                self.model_manager.unload_current_model()
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")

    
    def fill_missing_responses(self, prompt_batches: List[Dict[str, Any]], batch_size: int = 10) -> Dict[int, str]:
        """Fill missing responses for multiple prompts efficiently."""
        results = {}
        try:
            self.ensure_model_loaded()
            
            for i in range(0, len(prompt_batches), batch_size):
                batch = prompt_batches[i:i + batch_size]
                responses = self.generate_batch([p['prompt'] for p in batch])
                
                for prompt_data, response in zip(batch, responses):
                    if response:  
                        results[prompt_data['id']] = response
                        self.logger.info(f"Generated response for prompt {prompt_data['id']}")
                        
            return results
            
        except Exception as e:
            self.logger.error(f"Error filling missing responses: {str(e)}")
            return results
        
                
    def __del__(self):
        self.cleanup()
