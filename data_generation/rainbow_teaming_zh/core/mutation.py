

from typing import Dict, Optional, Any, List
import logging
from models.mutator_llm import MutatorLLM
import re
from core.database import ArchiveDB
from datetime import datetime
import time
from config.constants import MUTATION_TYPES, MODEL_CONFIG
from models.sub_mutator_llm import SubMutatorLLM


class MutationOperator:
    """Mutation operator with balanced validation and cleaning process."""
    
    def __init__(self, mutator_llm: MutatorLLM, archive_db: ArchiveDB, 
                 sub_mutator_batch_size: int = 3):
        self.mutator_llm = mutator_llm
        self.archive_db = archive_db
        self.sub_mutator = SubMutatorLLM(**MODEL_CONFIG['SUB_MUTATOR_MODEL'])
        self.sub_mutator_batch_size = sub_mutator_batch_size
        self.max_retries = 3
        self.sub_mutator_loaded = False
        
        
        self.error_patterns = [
            r'error\s*:',
            r'failed\s*:',
            r'unable to process',
            r'mutation failed',
            r'invalid prompt'
        ]
        self.error_pattern = re.compile('|'.join(self.error_patterns), re.IGNORECASE)
        
        
        self.stats = {
            'total_attempts': 0,
            'successful_mutations': 0,
            'failed_mutations': 0,
            'llm_mutations': 0,
            'sub_mutator_mutations': 0,
            'sub_mutator_success_rate': 0.0
        }

    def _extract_prompt(self, raw_text: str) -> Optional[str]:
        """Extract the actual prompt from the generated text."""
        
        prefix_patterns = [
            r'^Refined Prompt:\s*["\']?(.*?)["\']?\s*$',
            r'^Generated Prompt:\s*["\']?(.*?)["\']?\s*$',
            r'^Mutated Prompt:\s*["\']?(.*?)["\']?\s*$',
            r'^["\']?(.*?)["\']?\s*$'
        ]
        
        for pattern in prefix_patterns:
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return raw_text.strip()


    def _clean_prompt(self, raw_prompt: str) -> Optional[str]:
        """Clean the prompt while preserving valid content."""
        try:
            if not raw_prompt:
                return None
                
            cleaned = self._extract_prompt(raw_prompt)
            
            if not cleaned:
                return None
                
            if len(cleaned.strip()) < 5:
                return None
                
            if self.error_pattern.search(cleaned):
                logging.debug(f"Found error message in prompt: {cleaned[:100]}")
                return None
                
            cleaned = re.sub(r'^["\'](.*)["\']\s*$', r'\1', cleaned)
            cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
            cleaned = cleaned.strip()
            
            return cleaned if cleaned else None
            
        except Exception as e:
            logging.error(f"Error in clean_prompt: {str(e)}")
            return None
        

    def mutate_batch(
        self,
        prompts: List[str],
        current_descriptors: List[Dict[str, str]],
        target_descriptors: List[Dict[str, str]],
        max_retries: int = 3
    ) -> List[Optional[Dict[str, Any]]]:
        """Mutate multiple prompts in batch with proper cleanup."""
        results = []
        
        try:
            logging.info(f"Starting batch mutation for {len(prompts)} prompts")
            self.mutator_llm.ensure_model_loaded()
            
            for prompt, current, target in zip(prompts, current_descriptors, target_descriptors):
                try:
                    result = self.mutate_prompt(
                        prompt=prompt,
                        current_descriptor=current,
                        target_descriptor=target,
                        max_retries=max_retries
                    )
                    
                    if result:
                        self.stats['llm_mutations'] += 1
                        
                    results.append(result)
                    
                except Exception as e:
                    logging.error(f"Batch mutation failed for prompt: {str(e)}")
                    results.append(None)
            
            return results
            
        finally:
            logging.info("Unloading mutator model after batch mutation")
            self.mutator_llm.unload_model()

    def mutate_prompt(
        self,
        prompt: str,
        current_descriptor: Dict[str, str],
        target_descriptor: Dict[str, str],
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        """Mutate a prompt using LLM with improved handling."""
        self.stats['total_attempts'] += 1
        
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1}/{max_retries}:")
                mutated_prompt = self.mutator_llm.generate_mutation(
                    prompt=prompt,
                    target_style=target_descriptor['style'],
                    target_category=target_descriptor['category']
                )
                
                if not mutated_prompt:
                    logging.warning("Mutator returned empty output")
                    continue
                
                
                cleaned_prompt = self._clean_prompt(mutated_prompt)
                
                if not cleaned_prompt:
                    logging.warning(f"Failed to clean prompt attempt {attempt + 1}")
                    continue
                
                self.stats['successful_mutations'] += 1
                return {
                    'prompt': cleaned_prompt,
                    'style': target_descriptor['style'],
                    'category': target_descriptor['category'],
                    'mutation_type': 'llm',
                    'metadata': {
                        'original_prompt': prompt,
                        'mutation_attempt': attempt + 1,
                        'mutation_path': [
                            {
                                'from_style': current_descriptor['style'],
                                'from_category': current_descriptor['category'],
                                'to_style': target_descriptor['style'],
                                'to_category': target_descriptor['category']
                            }
                        ]
                    }
                }
                
            except Exception as e:
                logging.error(f"Mutation attempt {attempt + 1} failed: {str(e)}")
                continue
        
        self.stats['failed_mutations'] += 1
        return None
        
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive mutation statistics."""
        total_mutations = max(1, self.stats['llm_mutations'] + self.stats['sub_mutator_mutations'])
        total_attempts = max(1, self.stats['total_attempts'])
        
        return {
            **self.stats,
            'overall_success_rate': self.stats['successful_mutations'] / total_attempts,
            'llm_success_rate': self.stats['llm_mutations'] / total_mutations,
            'sub_mutator_success_rate': self.stats['sub_mutator_mutations'] / total_mutations,
            'mutation_type_distribution': {
                'llm': self.stats['llm_mutations'],
                'sub_mutator': self.stats['sub_mutator_mutations']
            }
        }

    def process_effective_prompt(self, 
                            prompt_data: Dict[str, Any],
                            effectiveness_score: float) -> List[Dict[str, Any]]:
        """Process effective prompts using sub-mutator LLM."""
        try:
            logging.info(f"Starting sub-mutator processing for effective prompt {prompt_data['id']}")
            if not self.sub_mutator_loaded:
                self.sub_mutator.ensure_model_loaded()
                self.sub_mutator_loaded = True
                logging.info("Sub-mutator model loaded")
            
            current_descriptor = {
                'style': prompt_data['attack_style'],
                'category': prompt_data['attack_category']
            }
            
            target_descriptor = current_descriptor.copy()
            
            logging.info(f"Generating {self.sub_mutator_batch_size} sub-mutator variants")
            results = self.sub_mutator.mutate_batch(
                prompts=[prompt_data['prompt']] * self.sub_mutator_batch_size,
                current_descriptors=[current_descriptor] * self.sub_mutator_batch_size,
                target_descriptors=[target_descriptor] * self.sub_mutator_batch_size
            )
            
            
            valid_results = [r for r in results if r is not None]
            
            if valid_results:
                logging.info(f"Successfully generated {len(valid_results)}/{self.sub_mutator_batch_size} sub-mutator variants")
                self.stats['sub_mutator_mutations'] += len(valid_results)
                
                
                stored_variant_ids = []
                for idx, variant in enumerate(valid_results, 1):
                    try:
                        variant_data = {
                            'prompt': variant['prompt'],
                            'attack_style': variant['style'],
                            'attack_category': variant['category'],
                            'llama_guard_score': prompt_data.get('llama_guard_score', 0.7),
                            'fitness_score': effectiveness_score * 0.95,
                            'target_response': None,  
                            'component_scores': prompt_data.get('component_scores', {}),
                            'parent_id': prompt_data['id'],
                            'mutation_type': MUTATION_TYPES['SUB_MUTATOR'],
                            'is_effective': True,
                            'metadata': {
                                'sub_mutator_score': effectiveness_score,
                                'parent_effectiveness': effectiveness_score,
                                'generation_timestamp': datetime.now().isoformat(),
                                'mutation_path': variant['metadata'].get('mutation_path', []) + [{
                                    'type': MUTATION_TYPES['SUB_MUTATOR'],
                                    'parent_id': prompt_data['id']
                                }]
                            }
                        }
                        
                        variant_id = self.archive_db.add_prompt(**variant_data)
                        if variant_id:
                            stored_variant_ids.append(variant_id)
                            self.stats['successful_mutations'] += 1
                            logging.info(f"Stored sub-mutator variant {idx}/{len(valid_results)} (ID: {variant_id})")
                            
                    except Exception as e:
                        logging.error(f"Error storing sub-mutator variant {idx}: {str(e)}")
                        continue
                
                logging.info(f"Successfully stored {len(stored_variant_ids)} sub-mutator variants")
                return stored_variant_ids
            else:
                logging.warning("No valid sub-mutator variants generated")
                return []
            
        except Exception as e:
            logging.error(f"Sub-mutator processing failed: {str(e)}")
            return []
        
    def cleanup_sub_mutator(self):
        """Cleanup sub-mutator only at end of iteration if it was loaded."""
        try:
            if self.sub_mutator_loaded:
                logging.info("Cleaning up sub-mutator at end of iteration")
                self.sub_mutator.cleanup()
                self.sub_mutator_loaded = False
                logging.info("Sub-mutator successfully unloaded")
        except Exception as e:
            logging.error(f"Error during sub-mutator cleanup: {str(e)}")