

import torch
import logging
from typing import Optional, Dict, Any, Literal
from pathlib import Path
from llama_cpp import Llama
import gc
import psutil
import time

from config.constants import MODEL_CONFIG
from config.settings import VRAM_LIMIT_GB, SYSTEM_RAM_LIMIT_GB, PRIMARY_DEVICE

ModelRole = Literal["target", "mutator", "judge"]

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        
        self.current_model = None
        self.current_role: Optional[ModelRole] = None
        self.current_model_path = None
        
        
        self.configs_cache = {}
        self.ram_cached_models = {}
        
        
        self.vram_limit = int(VRAM_LIMIT_GB * 1024 * 1024 * 1024)  
        self.system_ram_limit = int(SYSTEM_RAM_LIMIT_GB * 1024 * 1024 * 1024)
        self.ram_threshold = min(
            int(self.system_ram_limit * 0.9),  
            int(psutil.virtual_memory().total * 0.8)  
        )
        
        
        self._setup_memory_environment()
        self._initialized = True



    def _setup_memory_environment(self):
        """Setup optimized CUDA environment and memory monitoring."""
        try:
            if torch.cuda.is_available():
                
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                
                
                self.total_vram = torch.cuda.get_device_properties(0).total_memory
                self.available_vram = min(
                    self.vram_limit,
                    self.total_vram - (2 * 1024 * 1024 * 1024)  
                )
                
                
                memory_fraction = min(0.95, self.vram_limit / self.total_vram)
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                
                logging.info(f"GPU detected with {self.total_vram/1024**3:.2f}GB total VRAM")
                logging.info(f"VRAM limit set to {VRAM_LIMIT_GB:.2f}GB")
                
            self.total_ram = psutil.virtual_memory().total
            logging.info(f"System RAM: {self.total_ram/1024**3:.2f}GB")
            logging.info(f"RAM limit set to {SYSTEM_RAM_LIMIT_GB:.2f}GB")
        except Exception as e:
            logging.error(f"Memory environment setup failed: {str(e)}")
            raise

    def _cache_model_to_ram(self, role: ModelRole, model: Any, config: Dict):
        """Cache model to RAM with memory limits."""
        try:
            if not hasattr(self, 'ram_cached_models'):
                self.ram_cached_models = {}
                
            available_ram = psutil.virtual_memory().available
            if available_ram < self.ram_threshold:
                self._clear_ram_cache(exclude_role=role)
            
            self.ram_cached_models[role] = {
                "model": model,
                "config": config
            }
            
        except Exception as e:
            logging.error(f"Model caching failed for role {role}: {str(e)}")
            self._clear_ram_cache()  

    def _clear_ram_cache(self, exclude_role: Optional[ModelRole] = None):
        """Clear RAM cache with safety checks."""
        try:
            if not hasattr(self, 'ram_cached_models'):
                self.ram_cached_models = {}
                return
                
            roles_to_clear = [role for role in self.ram_cached_models if role != exclude_role]
            for role in roles_to_clear:
                if role in self.ram_cached_models:
                    model_data = self.ram_cached_models.pop(role, None)
                    if model_data and model_data.get("model") is not None:
                        del model_data["model"]
                        gc.collect()
                        
        except Exception as e:
            logging.error(f"RAM cache clearing failed: {str(e)}")
            
            self.ram_cached_models = {}
            gc.collect()


    def load_model_for_role(self, role: ModelRole, model_path: str, config: Optional[Dict] = None) -> Optional[Llama]:
        try:
            
            if self.current_role == role and self.current_model_path == model_path and self.current_model is not None:
                logging.info(f"Model already loaded for role {role}")
                return self.current_model

            logging.info(f"Loading model for role: {role}")

            self._ensure_memory_available()

            if config is None:
                config = self._get_role_config(role)
            config.pop('model_path', None)

            self.current_model = Llama(
                model_path=str(model_path),
                **config
            )

            self.current_role = role
            self.current_model_path = model_path
            self.configs_cache[role] = config

            logging.info(f"Successfully loaded model for role {role}")
            return self.current_model

        except Exception as e:
            logging.error(f"Model loading failed for role {role}: {str(e)}")
            self._ensure_memory_available()
            raise


    def unload_current_model(self):
        """Completely unload current model and clear memory."""
        try:
            if self.current_model is not None:
                logging.info(f"Unloading model for role: {self.current_role}")
                
                
                self.current_model.close()  
                
                
                del self.current_model
                self.current_model = None
                self.current_role = None
                self.current_model_path = None
                
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    
                    
                    
                    
                    for _ in range(3):
                        gc.collect()
                        time.sleep(1)  
                    
                    

                    
                
                gc.collect()
                
                
        except Exception as e:
            logging.error(f"Model unloading failed: {str(e)}")
            raise

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.unload_current_model()
        except:
            pass

    def _ensure_memory_available(self):
        """Aggressive memory cleanup before loading new model."""
        try:
            if torch.cuda.is_available():
                
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                
                
                if self.current_model is not None:
                    del self.current_model
                    self.current_model = None
                    self.current_role = None
                    self.current_model_path = None
                
                
                gc.collect()
                torch.cuda.synchronize()
                
        except Exception as e:
            logging.error(f"Memory cleanup failed: {str(e)}")
            raise

    def _get_role_config(self, role: ModelRole) -> Dict:
        """Get optimized configuration based on role."""
        if role not in ["target", "mutator", "judge"]:
            raise ValueError(f"Invalid role: {role}")

        base_config = MODEL_CONFIG[f'{role.upper()}_MODEL'].copy()
        base_config.pop('model_path', None)

        return base_config
    

    def _ensure_memory_available(self):
        """Aggressive memory cleanup before loading new model."""
        try:
            if torch.cuda.is_available():
                
                if self.current_model is not None:
                    self.unload_current_model()
                
                
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
        except Exception as e:
            logging.error(f"Memory cleanup failed: {str(e)}")
            raise