

from typing import Dict, Any, List, Optional, Tuple
import random
import numpy as np
from sqlalchemy import func  
from config.settings import SAMPLING_TEMPERATURE, LOW_FITNESS_BIAS
from config.constants import (
    ATTACK_STYLES, 
    ATTACK_CATEGORIES, 
    MODEL_CONFIG, 
    MUTATION_TYPES
)
import logging
from core.database import Prompt

class Sampler:
    def __init__(self, archive_db):
        self.archive = archive_db
        self.temperature = SAMPLING_TEMPERATURE
        self.low_fitness_bias = LOW_FITNESS_BIAS
        self.logger = logging.getLogger(__name__)
        self.mutation_weights = MODEL_CONFIG['SAMPLER']['mutation_weights']
        self.strategy_weights = MODEL_CONFIG['SAMPLER']['strategy_weights']

    def _get_random_prompt(self) -> Dict[str, Any]:
        """Fallback method to get any random prompt from the database."""
        try:
            if self.archive.session.in_transaction():
                self.archive.session.commit() 
                
            
            prompt_query = self.archive.session.query(Prompt.id, Prompt.prompt).filter(
                func.length(Prompt.prompt) <= MODEL_CONFIG['SAMPLER']['max_prompt_length']
            )
            prompts = prompt_query.all()
                
            if not prompts:
                self.logger.warning("No prompts found within length limit")
                raise ValueError("No prompts found in database")
                
            
            selected_prompt = random.choice(prompts)
            self.logger.info(f"Sampled prompt [ID: {selected_prompt[0]}, Length: {len(selected_prompt[1])}]")
            
            
            prompt_data = self.archive.get_prompt_by_id(selected_prompt[0])
            if prompt_data:
                return prompt_data
                
            raise ValueError(f"Could not retrieve prompt with ID: {selected_prompt[0]}")
            
        except Exception as e:
            self.logger.error(f"Error in random prompt retrieval: {str(e)}")
            return {
                'id': 0,
                'prompt': "Default fallback prompt",
                'attack_style': random.choice(ATTACK_STYLES),
                'attack_category': random.choice(ATTACK_CATEGORIES),
                'llama_guard_score': 0.7,
                'fitness_score': 0.5,
                'target_response': "",
                'metadata': {'fallback': True}
            }
    def sample_prompt(self, 
                    strategy: Optional[str] = None, 
                    min_fitness: Optional[float] = None,
                    mutation_type: Optional[str] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Enhanced prompt sampling with proper strategy handling and empty response tracking."""
        try:
            if self.archive.session.in_transaction():
                self.archive.session.commit()
                
            
            effective_count = self.archive.session.query(Prompt).filter(
                Prompt.is_effective == True,
                func.length(Prompt.prompt) <= MODEL_CONFIG['SAMPLER']['max_prompt_length']
            ).count()
            
            self.logger.info(f"Found {effective_count} effective prompts within length limit")
            
            
            available_strategies = []
            weights = []
            
            if effective_count > 0:
                available_strategies.extend(['effective', 'novel', 'random'])
                weights.extend([
                    self.strategy_weights['effective'],
                    self.strategy_weights['novel'], 
                    self.strategy_weights['random']
                ])
            else:
                available_strategies.extend(['novel', 'random'])
                
                total_weight = self.strategy_weights['novel'] + self.strategy_weights['random']
                weights.extend([
                    self.strategy_weights['novel'] / total_weight,
                    self.strategy_weights['random'] / total_weight
                ])
            
            if strategy is None:
                
                total_weight = sum(weights)
                weights = [w/total_weight for w in weights]
                strategy = np.random.choice(available_strategies, p=weights)
                self.logger.info(f"Selected strategy: {strategy} (available: {available_strategies}, weights: {weights})")
                
            prompt_data = None
            empty_response_prompts = []
            
            
            if strategy == 'effective':
                prompt_data = self._sample_effective_prompt(
                    min_fitness=min_fitness,
                    mutation_type=mutation_type
                )
            elif strategy == 'novel':
                prompt_data = self._sample_novel_prompt(
                    min_fitness=min_fitness,
                    mutation_type=mutation_type
                )
            else:  
                prompt_data = self._sample_random_prompt(
                    min_fitness=min_fitness,
                    mutation_type=mutation_type
                )
                
            if prompt_data:
                self.logger.info(
                    f"Sampled prompt [ID: {prompt_data.get('id', 'N/A')}, "
                    f"Length: {len(prompt_data.get('prompt', ''))}, "
                    f"Strategy: {strategy}, "
                    f"Is_effective: {prompt_data.get('is_effective', False)}]"
                )
                
                
                if prompt_data.get('mutation_type') == MUTATION_TYPES['SUB_MUTATOR']:
                    if not prompt_data.get('target_response'):
                        empty_response_prompts.append(prompt_data)
                        self.logger.info(f"Found empty response in sampled prompt {prompt_data['id']}")
                    
                    
                    if prompt_data.get('parent_id'):
                        parent = self.archive.get_prompt_by_id(prompt_data['parent_id'])
                        if parent and not parent.get('target_response'):
                            empty_response_prompts.append(parent)
                            self.logger.info(f"Found empty response in parent prompt {parent['id']}")
                
                return prompt_data, empty_response_prompts
                
            raise ValueError(f"No prompt found using strategy: {strategy}")
            
        except Exception as e:
            self.logger.error(f"Error in sample_prompt: {str(e)}")
            return self._get_random_prompt(), []
    
    def _sample_effective_prompt(self, 
                            min_fitness: Optional[float] = None,
                            mutation_type: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced effective prompt sampling with better error handling."""
        try:
            
            query = self.archive.session.query(Prompt).filter(
                Prompt.is_effective == True,
                func.length(Prompt.prompt) <= MODEL_CONFIG['SAMPLER']['max_prompt_length']
            )
            
            if min_fitness is not None:
                query = query.filter(Prompt.fitness_score >= min_fitness)
            if mutation_type:
                query = query.filter(Prompt.mutation_type == mutation_type)
                
            
            matching_count = query.count()
            if matching_count == 0:
                self.logger.warning("No effective prompts found matching criteria")
                return None
                
            
            prompts = query.all()
            fitness_scores = np.array([p.fitness_score for p in prompts])
            probs = np.exp(fitness_scores / self.temperature)
            probs = probs / probs.sum()
            
            selected = np.random.choice(prompts, p=probs)
            
            return {
                'id': selected.id,
                'prompt': selected.prompt,
                'attack_style': selected.attack_style,
                'attack_category': selected.attack_category,
                'llama_guard_score': selected.llama_guard_score,
                'fitness_score': selected.fitness_score,
                'target_response': selected.target_response,
                'is_effective': selected.is_effective,
                'effectiveness_score': selected.effectiveness_score,
                'metadata': selected.extra_data,
                'parent_id': selected.parent_id,
                'mutation_path': selected.mutation_path,
                'generation': selected.generation,
                'mutation_type': selected.mutation_type
            }
            
        except Exception as e:
            self.logger.error(f"Error sampling effective prompt: {str(e)}")
            return None
            
    def _sample_novel_prompt(self, 
                            min_fitness: Optional[float] = None,
                            mutation_type: Optional[str] = None) -> Dict[str, Any]:
        """Sample from recent generations with fallback."""
        try:
            recent_prompts = self.archive.get_all_prompts(
                effective_only=False,
                min_fitness=min_fitness,
                mutation_type=mutation_type,
                max_length=MODEL_CONFIG['SAMPLER']['max_prompt_length'],
                limit=50
            )
            
            if not recent_prompts:
                return self._get_random_prompt()
            
            scores = [self._calculate_novelty_score(prompt) for prompt in recent_prompts]
            total_score = sum(scores)
            
            if total_score == 0:
                selected_prompt = random.choice(recent_prompts)
                self.logger.info(f"Sampled novel prompt randomly [ID: {selected_prompt.get('id', 'N/A')}, "
                               f"Length: {len(selected_prompt.get('prompt', ''))}]")
                return selected_prompt
            
            probabilities = [score/total_score for score in scores]
            idx = np.random.choice(len(recent_prompts), p=probabilities)
            selected_prompt = recent_prompts[idx]
            self.logger.info(f"Sampled novel prompt [ID: {selected_prompt.get('id', 'N/A')}, "
                           f"Length: {len(selected_prompt.get('prompt', ''))}]")
            return selected_prompt
            
        except Exception as e:
            self.logger.error(f"Novel sampling failed: {str(e)}")
            return self._get_random_prompt()
    
    def _sample_random_prompt(self, 
                        min_fitness: Optional[float] = None,
                        mutation_type: Optional[str] = None) -> Dict[str, Any]:
        """Random sampling with fallback."""
        try:
            prompt = self.archive.sample_prompt(
                temperature=1.0,
                low_fitness_bias=0.5,
                effective_only=False,
                min_fitness=min_fitness,
                mutation_type=mutation_type,
                max_length=MODEL_CONFIG['SAMPLER']['max_prompt_length']
            )
            if prompt:
                self.logger.info(f"Sampled random prompt [ID: {prompt.get('id', 'N/A')}, "
                               f"Length: {len(prompt.get('prompt', ''))}]")
            return prompt
        except Exception:
            return self._get_random_prompt()
        
    def sample_target_descriptors(self, 
                                source_prompt: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Sample target descriptors for mutation, with option to consider source prompt.
        
        Args:
            source_prompt: Optional source prompt to influence sampling
        Returns:
            Dict with 'style' and 'category' keys
        """
        try:
            if self.archive.session.in_transaction():
                self.archive.session.commit()  
                
            stats = self.archive.get_grid_statistics()
            style_dist = stats.get('style_distribution', {})
            category_dist = stats.get('category_distribution', {})
            
            if source_prompt:
                
                current_style = source_prompt.get('attack_style')
                current_category = source_prompt.get('attack_category')
                
                if current_style and current_style in style_dist:
                    style_dist[current_style] *= 0.5  
                if current_category and current_category in category_dist:
                    category_dist[current_category] *= 0.5  
            
            
            if not style_dist or not category_dist:
                return {
                    'style': random.choice(ATTACK_STYLES),
                    'category': random.choice(ATTACK_CATEGORIES)
                }
            
            return {
                'style': self._sample_with_bias(style_dist),
                'category': self._sample_with_bias(category_dist)
            }
            
        except Exception as e:
            self.logger.error(f"Error sampling target descriptors: {str(e)}")
            
            return {
                'style': random.choice(ATTACK_STYLES),
                'category': random.choice(ATTACK_CATEGORIES)
            }

    def _sample_with_bias(self, distribution: Dict[str, float]) -> str:
        """Enhanced distribution sampling with better error handling."""
        try:
            
            if not distribution or sum(distribution.values()) == 0:
                return random.choice(list(distribution.keys()))
            
            
            weights = {k: 1.0 - v for k, v in distribution.items()}
            total = sum(weights.values())
            
            if total == 0:
                return random.choice(list(distribution.keys()))
            
            
            weights = {k: pow(v/total, 1.0/self.temperature) for k, v in weights.items()}
            
            
            items = np.array(list(weights.keys()))
            probs = np.array(list(weights.values()))
            probs = probs / np.sum(probs)  
            
            return np.random.choice(items, p=probs)
            
        except Exception as e:
            self.logger.error(f"Biased sampling failed: {str(e)}")
            if distribution:
                return random.choice(list(distribution.keys()))
            return random.choice(ATTACK_STYLES)  
        
    def _handle_deferred_response(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prompts with deferred response generation."""
        if prompt_data.get('mutation_type') == MUTATION_TYPES['SUB_MUTATOR'] and not prompt_data.get('target_response'):
            self.logger.info(f"Found sub-mutator prompt {prompt_data['id']} without response")
            return {
                **prompt_data,
                'needs_response': True
            }
        return prompt_data
    
    def _calculate_novelty_score(self, prompt: Dict[str, Any]) -> float:
        """Calculate novelty score for a prompt."""
        try:
            
            base_score = prompt.get('fitness_score', 0.0)
            generation = prompt.get('generation', 0)
            
            
            generation_bonus = min(generation * 0.1, 0.5)
            
            
            effectiveness_bonus = 0.3 if prompt.get('is_effective', False) else 0.0
            
            
            novelty_score = (base_score + generation_bonus + effectiveness_bonus)
            
            return max(0.0, min(novelty_score, 1.0))  
            
        except Exception as e:
            self.logger.error(f"Error calculating novelty score: {str(e)}")
            return 0.0