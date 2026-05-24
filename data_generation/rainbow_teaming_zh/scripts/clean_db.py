import sys
import os
from pathlib import Path
import json
import logging
from tqdm import tqdm
import torch
from datetime import datetime
import time
from llama_cpp import Llama


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.constants import (
    MODEL_CONFIG,
    PROMPT_CLEANING_TEMPLATE,
    CLEANING_PROGRESS_FILE,
    CLEANING_BATCH_SIZE,
    CLEANING_CHECKPOINTS
)
from core.database import ArchiveDB, Prompt
from core.model_manager import ModelManager
from utils.logging import setup_logging
from typing import Optional
from config.settings import DB_PATH

DEFAULT_DB_PATH = DB_PATH

class PromptCleaner:
    def __init__(self, db_path: str = None):
        """Initialize the prompt cleaner with database connection and model."""
        self.logger = logging.getLogger(__name__)
        
        
        if db_path is None:
            db_path = os.getenv('DB_PATH', DEFAULT_DB_PATH)
        
        
        self.db_path = Path(db_path).resolve()
        
        
        progress_dir = Path(CLEANING_PROGRESS_FILE).parent
        progress_dir.mkdir(parents=True, exist_ok=True)
        
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.archive_db = ArchiveDB(db_path=str(self.db_path))
        self.model_manager = ModelManager()
        self.progress = self._load_progress()

    def _create_default_progress(self) -> dict:
        """Create default progress state with relative path."""
        return {
            'last_processed_id': 0,
            'cleaned_count': 0,
            'timestamp': None,
            'db_relative_path': self._get_relative_path(self.db_path),
            'last_prompt': None,
            'db_root': str(project_root)  
        }

    def _get_relative_path(self, path: Path) -> str:
        """Convert absolute path to relative path from project root."""
        try:
            return str(path.relative_to(project_root))
        except ValueError:
            
            return str(path)

    def _reconstruct_db_path(self, progress: dict) -> Path:
        """Reconstruct absolute DB path from stored relative path."""
        db_root = Path(progress.get('db_root', project_root))
        relative_path = progress.get('db_relative_path')
        
        if not relative_path:
            return self.db_path
            
        
        reconstructed_path = db_root / relative_path
        
        
        return reconstructed_path if reconstructed_path.exists() else self.db_path

    def _load_progress_file(self) -> Optional[dict]:
        """Load progress from file with error handling."""
        try:
            if not os.path.exists(CLEANING_PROGRESS_FILE):
                return None
                
            with open(CLEANING_PROGRESS_FILE, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                
            if not isinstance(progress, dict):
                self.logger.error("Progress file contains invalid data")
                return None
                
            required_keys = ['last_processed_id', 'cleaned_count', 'db_relative_path']
            if not all(key in progress for key in required_keys):
                self.logger.error("Progress file missing required data")
                return None
                
            return progress
            
        except json.JSONDecodeError:
            self.logger.error("Progress file contains invalid JSON")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error loading progress file: {str(e)}")
            return None
    def _load_progress(self) -> dict:
        """Load cleaning progress with verification."""
        try:
            progress = self._load_progress_file()
            if progress and self._verify_progress(progress):
                self.logger.info(f"Resuming from prompt ID {progress['last_processed_id']}")
                
                
                return {
                    'last_processed_id': int(progress['last_processed_id']),
                    'cleaned_count': int(progress['cleaned_count']),
                    'timestamp': progress.get('timestamp'),
                    'db_relative_path': progress.get('db_relative_path', self._get_relative_path(self.db_path)),
                    'db_root': progress.get('db_root', str(project_root)),
                    'last_prompt': progress.get('last_prompt')
                }

            if os.path.exists(CLEANING_PROGRESS_FILE):
                self.logger.warning("Invalid progress file found. Creating backup and starting fresh.")
                backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_file = f"{CLEANING_PROGRESS_FILE}.{backup_time}.bak"
                
                try:
                    os.rename(CLEANING_PROGRESS_FILE, backup_file)
                except Exception as e:
                    self.logger.error(f"Failed to create backup of progress file: {e}")
                
            return self._create_default_progress()
            
        except Exception as e:
            self.logger.error(f"Error loading progress, starting fresh: {str(e)}")
            return self._create_default_progress()

    def _verify_progress(self, progress: dict) -> bool:
        """Verify progress data consistency using relative paths."""
        try:
            
            stored_db_path = self._reconstruct_db_path(progress)
            current_db_path = self.db_path.resolve()
            
            if stored_db_path.resolve() != current_db_path:
                self.logger.warning(
                    f"Progress file is for a different database.\n"
                    f"Expected: {current_db_path}\nFound: {stored_db_path}"
                )
                return False

            
            last_id = progress['last_processed_id']
            if last_id > 0:
                last_prompt = (
                    self.archive_db.session.query(Prompt)
                    .filter(Prompt.id == last_id)
                    .first()
                )
                
                if not last_prompt:
                    self.logger.error(f"Could not find prompt with ID {last_id}")
                    return False

            
            if not isinstance(progress['cleaned_count'], int):
                self.logger.error("Invalid cleaned_count type")
                return False

            if progress['cleaned_count'] < 0:
                self.logger.error("Invalid cleaned_count value")
                return False

            self.logger.info(
                f"Successfully verified progress. Last ID: {last_id}, "
                f"Cleaned Count: {progress['cleaned_count']}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Progress verification failed: {str(e)}")
            return False

    def _save_progress(self, last_id: int, cleaned_count: int):
        """Save current cleaning progress using relative paths."""
        backup_file = str(Path(CLEANING_PROGRESS_FILE).with_suffix('.backup'))
        try:
            progress = {
                'last_processed_id': last_id,
                'cleaned_count': cleaned_count,
                'timestamp': datetime.now().isoformat(),
                'db_relative_path': self._get_relative_path(self.db_path),
                'db_root': str(project_root),
                'last_prompt': None
            }
            
            
            last_prompt = self.archive_db.session.query(Prompt).get(last_id)
            if last_prompt:
                progress['last_prompt'] = hash(last_prompt.prompt)
            
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
            
            
            with open(CLEANING_PROGRESS_FILE, 'w', encoding='utf-8') as f:
                json.dump(progress, f, indent=2, ensure_ascii=False)
                
            
            if os.path.exists(backup_file):
                os.remove(backup_file)
                
            self.logger.info(f"Progress saved successfully. Last ID: {last_id}, Cleaned: {cleaned_count}")
                
        except Exception as e:
            self.logger.error(f"Error saving progress: {str(e)}")
            if os.path.exists(backup_file):
                self.logger.info("Restoring progress from backup")
                try:
                    with open(backup_file, 'r', encoding='utf-8') as f:
                        backup_progress = json.load(f)
                    with open(CLEANING_PROGRESS_FILE, 'w', encoding='utf-8') as f:
                        json.dump(backup_progress, f, indent=2, ensure_ascii=False)
                except Exception as restore_error:
                    self.logger.error(f"Failed to restore from backup: {str(restore_error)}")

    def clean_prompt(self, prompt: str, style: str, category: str) -> str:
        """Clean a single prompt using the cleaning model."""
        try:
            cleaning_prompt = PROMPT_CLEANING_TEMPLATE.format(
                prompt=prompt,
                style=style,
                category=category
            )
            
            messages = [
                {"role": "system", "content": "你是一個專門清理對抗性提示的助手。"},
                {"role": "user", "content": cleaning_prompt}
            ]

            
            response_iter = self.model.create_chat_completion(
                messages=messages,
                temperature=MODEL_CONFIG['CLEANING_MODEL_CONFIG']['temperature'],
                top_p=MODEL_CONFIG['CLEANING_MODEL_CONFIG']['top_p'],
                top_k=MODEL_CONFIG['CLEANING_MODEL_CONFIG']['top_k'],
                max_tokens=MODEL_CONFIG['CLEANING_MODEL_CONFIG']['max_tokens'],
                repeat_penalty=MODEL_CONFIG['CLEANING_MODEL_CONFIG']['repeat_penalty'],
                stream=True
            )

            cleaned_prompt = ""
            for response in response_iter:
                if 'choices' in response and response['choices']:
                    delta = response['choices'][0]['delta']
                    if 'content' in delta:
                        content = delta['content']
                        print(content, end='', flush=True)
                        cleaned_prompt += content

            print()  
            cleaned = cleaned_prompt.strip()

            
            if len(cleaned) < 10 or '[placeholder]' in cleaned:
                self.logger.warning("Cleaning validation failed. Returning original prompt.")
                return prompt

            return cleaned

        except Exception as e:
            self.logger.error(f"Error cleaning prompt: {str(e)}")
            return prompt

    def process_prompts(self):
        """Process all prompts in the database, continuing from last progress."""
        try:
            
            self.model = self.model_manager.load_model_for_role(
                "cleaning",
                MODEL_CONFIG['CLEANING_MODEL_CONFIG']['model_path'],
                MODEL_CONFIG['CLEANING_MODEL_CONFIG']
            )

            
            total_prompts = (
                self.archive_db.session.query(Prompt.id)
                .filter(Prompt.id > self.progress['last_processed_id'])
                .count()
            )
            
            if total_prompts == 0:
                self.logger.info("No new prompts to clean")
                return

            self.logger.info(f"Starting cleaning from ID {self.progress['last_processed_id']}")
            self.logger.info(f"Found {total_prompts} prompts to process")

            cleaned_count = self.progress['cleaned_count']
            last_id = self.progress['last_processed_id']
            
            
            with tqdm(total=total_prompts, desc="Processing prompts") as pbar:
                while True:
                    
                    batch = (
                        self.archive_db.session.query(Prompt)
                        .filter(Prompt.id > last_id)
                        .order_by(Prompt.id)
                        .limit(CLEANING_BATCH_SIZE)
                        .all()
                    )
                    
                    if not batch:
                        break
                    
                    for prompt in batch:
                        try:
                            original_text = prompt.prompt
                            cleaned_text = self.clean_prompt(
                                prompt.prompt,
                                prompt.attack_style,
                                prompt.attack_category
                            )
                            
                            if cleaned_text != original_text:
                                
                                with self.archive_db.transaction():
                                    prompt.prompt = cleaned_text
                                    if not prompt.extra_data:
                                        prompt.extra_data = {}
                                    prompt.extra_data.update({
                                        'cleaning_timestamp': datetime.now().isoformat(),
                                        'original_prompt': original_text
                                    })
                                cleaned_count += 1
                                self.logger.info(f"Cleaned prompt {prompt.id}")
                            
                            last_id = prompt.id
                            pbar.update(1)
                            
                        except Exception as e:
                            self.logger.error(f"Error processing prompt {prompt.id}: {str(e)}")
                            continue

                    
                    if last_id % CLEANING_CHECKPOINTS == 0:
                        self._save_progress(last_id, cleaned_count)
                    
                    
                    self.archive_db.session.commit()

            
            self._save_progress(last_id, cleaned_count)
            self.logger.info(f"Cleaning complete. Processed {cleaned_count} prompts.")

        except Exception as e:
            self.logger.error(f"Error in prompt cleaning process: {str(e)}")
            raise
        finally:
            if self.model:
                self.model_manager.unload_current_model()

def main():
    

    
    log_dir = Path("data/logs/cleaning")
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir)
    
    try:
        
        print(f"Current working directory: {os.getcwd()}")
        print(f"Project root: {project_root}")
        print(f"Default DB path: {DEFAULT_DB_PATH}")
        
        
        db_path_env = os.getenv('DB_PATH')
        print(f"DB_PATH environment variable: {db_path_env}")
        
        
        db_dir = Path(DEFAULT_DB_PATH).parent
        print(f"Database directory exists: {db_dir.exists()}")
        print(f"Database directory permissions: {oct(os.stat(db_dir).st_mode)[-3:]}")
        
        cleaner = PromptCleaner()
        print(f"Final resolved DB path: {cleaner.db_path}")
        print(f"DB file exists: {cleaner.db_path.exists()}")
        
        cleaner.process_prompts()
    except Exception as e:
        logging.error(f"Cleaning process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
