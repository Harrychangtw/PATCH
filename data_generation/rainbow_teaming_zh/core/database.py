
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Text, Boolean,
    Index, func, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.dialects.sqlite import JSON
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
import logging
from contextlib import contextmanager
import numpy as np
import sys

from config.settings import DB_PATH
from config.constants import ATTACK_STYLES, ATTACK_CATEGORIES, MUTATION_TYPES
from utils.fitness import EnhancedFitnessCalculator


if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

Base = declarative_base()

class Prompt(Base):
    """Stores prompt data and evaluation results with component scores."""
    __tablename__ = 'prompts'
    
    id = Column(Integer, primary_key=True)
    prompt = Column(Text, nullable=False)
    attack_style = Column(String(50), nullable=False)
    attack_category = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    mutation_type = Column(String(20), default='llm') 
    

    is_effective = Column(Boolean, default=False)
    effectiveness_score = Column(Float)
    

    llama_guard_score = Column(Float, nullable=False)
    fitness_score = Column(Float, nullable=False)
    target_response = Column(Text)

    safety_score = Column(Float)
    diversity_score = Column(Float)
    response_consistency = Column(Float)
    prompt_sophistication = Column(Float)
    category_alignment = Column(Float)
    judge_confidence = Column(Float)

    parent_id = Column(Integer, ForeignKey('prompts.id', ondelete='SET NULL'))
    parent_fitness = Column(Float)
    mutation_path = Column(JSON)
    generation = Column(Integer, default=0)
    

    extra_data = Column(JSON)
    
    __table_args__ = (
        Index('idx_prompt_features', attack_style, attack_category),
        Index('idx_prompt_fitness', fitness_score),
        Index('idx_prompt_effectiveness', is_effective),
        Index('idx_prompt_timestamp', timestamp),
        Index('idx_prompt_mutation', mutation_type),  
        Index('idx_component_scores', 
              safety_score, 
              diversity_score, 
              response_consistency, 
              prompt_sophistication, 
              category_alignment)
    )



class GridCell(Base):
    """Represents a cell in the MAP-Elites grid."""
    __tablename__ = 'grid_cells'
    
    id = Column(Integer, primary_key=True)
    style = Column(String(50), nullable=False)
    category = Column(String(50), nullable=False)
    current_fitness = Column(Float, default=0.0)
    update_count = Column(Integer, default=0)
    last_update = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    
    current_prompt_id = Column(Integer, ForeignKey('prompts.id', ondelete='SET NULL'))
    previous_prompt_id = Column(Integer, ForeignKey('prompts.id', ondelete='SET NULL'))
    
    current_prompt_id = Column(Integer, ForeignKey('prompts.id', ondelete='SET NULL'))
    current_prompt = relationship("Prompt", foreign_keys=[current_prompt_id])
    
    
    __table_args__ = (
        Index('idx_grid_location', style, category, unique=True),
        Index('idx_grid_fitness', current_fitness)
    )

class ArchiveDB:
    """Enhanced MAP-Elites archive implementation with database backend."""
    
    def __init__(self, db_path=None, temperature: float = 0.7, low_fitness_bias: float = 0.7):
        """Initialize the archive database with MAP-Elites parameters."""
        db_file = db_path or DB_PATH
        self.engine = create_engine(f'sqlite:///{db_file}', echo=False)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        self.temperature = temperature
        self.low_fitness_bias = low_fitness_bias
        
        
        self._initialize_grid()

    def _initialize_grid(self):
        """Create grid cells for all possible feature combinations."""
        with self.transaction():
            existing_cells = self.session.query(GridCell).count()
            if existing_cells == 0:
                for style in ATTACK_STYLES:
                    for category in ATTACK_CATEGORIES:
                        cell = GridCell(style=style, category=category)
                        self.session.add(cell)


    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        try:
            yield
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e

    def add_prompt(self, 
                  prompt: str, 
                  attack_style: str, 
                  attack_category: str,
                  llama_guard_score: float,
                  fitness_score: float,
                  target_response: Optional[str] = None,
                  component_scores: Optional[Dict[str, float]] = None,
                  parent_id: Optional[int] = None,
                  parent_fitness: Optional[float] = None,
                  mutation_path: Optional[List[Dict]] = None,
                  mutation_type: str = MUTATION_TYPES['LLM'],
                  is_effective: bool = False,
                  metadata: Optional[Dict[str, Any]] = None) -> int:
        """Enhanced prompt addition with sub-mutator support."""
        try:
            with self.transaction():
                
                if mutation_type not in [MUTATION_TYPES['LLM'], MUTATION_TYPES['SUB_MUTATOR']]:
                    raise ValueError(f"Invalid mutation type: {mutation_type}")
                
                
                generation = 0
                if parent_id and parent_fitness is None:
                    parent = self.session.query(Prompt).get(parent_id)
                    if parent:
                        parent_fitness = parent.fitness_score
                        generation = parent.generation + 1
                
                
                new_prompt = Prompt(
                    prompt=prompt,
                    attack_style=attack_style,
                    attack_category=attack_category,
                    mutation_type=mutation_type,
                    is_effective=is_effective,
                    effectiveness_score=fitness_score,
                    llama_guard_score=llama_guard_score,
                    fitness_score=fitness_score,
                    target_response=target_response,  
                    safety_score=component_scores.get('safety_score', 0.0),
                    diversity_score=component_scores.get('diversity_score', 0.0),
                    response_consistency=component_scores.get('response_consistency', 0.0),
                    prompt_sophistication=component_scores.get('prompt_sophistication', 0.0),
                    category_alignment=component_scores.get('category_alignment', 0.0),
                    judge_confidence=component_scores.get('judge_confidence', 0.0),
                    parent_id=parent_id,
                    parent_fitness=parent_fitness,
                    mutation_path=mutation_path,
                    generation=generation,
                    extra_data={
                        **(metadata or {}),
                        'needs_response': mutation_type == MUTATION_TYPES['SUB_MUTATOR'] and target_response is None
                    }
                )
                
                self.session.add(new_prompt)
                self.session.flush()
                
                logging.info(f"Added new prompt (ID: {new_prompt.id}, Type: {mutation_type}, "
                           f"Effective: {is_effective}, Needs Response: {target_response is None})")
                
                
                if is_effective and (target_response is not None or mutation_type != MUTATION_TYPES['SUB_MUTATOR']):
                    cell = self.session.query(GridCell).filter_by(
                        style=attack_style,
                        category=attack_category
                    ).first()
                    
                    if cell and fitness_score > (cell.current_fitness or 0.0):
                        self._update_grid_cell(new_prompt)
                
                return new_prompt.id
                
        except Exception as e:
            logging.error(f"Error adding prompt: {str(e)}")
            raise
            

    def sample_prompt(self, 
                    temperature: float = 0.7, 
                    low_fitness_bias: float = 0.7,
                    effective_only: bool = False,
                    min_fitness: Optional[float] = None,
                    mutation_type: Optional[str] = None,
                    max_length: Optional[int] = None) -> Dict[str, Any]:
        """Enhanced prompt sampling with fixed query building."""
        try:
            
            query = self.session.query(Prompt)
            
            
            if effective_only:
                query = query.filter(Prompt.is_effective == True)
            if min_fitness is not None:
                query = query.filter(Prompt.fitness_score >= min_fitness)
            if mutation_type is not None:
                query = query.filter(Prompt.mutation_type == mutation_type)
            if max_length is not None:
                query = query.filter(func.length(Prompt.prompt) <= max_length)
                
            
            prompts = query.all()
            if not prompts:
                raise ValueError("No matching prompts in archive")
            
            
            fitness_scores = np.array([p.fitness_score for p in prompts])
            if low_fitness_bias > 0:
                
                weights = 1 - (fitness_scores / fitness_scores.max()) if fitness_scores.max() > 0 else np.ones_like(fitness_scores)
            else:
                
                weights = fitness_scores / fitness_scores.max() if fitness_scores.max() > 0 else np.ones_like(fitness_scores)
                
            
            weights = np.exp(weights / temperature)
            probs = weights / weights.sum()
            
            
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
            logging.error(f"Error sampling prompt: {str(e)}")
            raise
    def sample_target_descriptors(self) -> Dict[str, str]:
        """Sample target descriptors with bias towards low-performing areas."""
        style_fitness = {}
        category_fitness = {}
        
        for style in ATTACK_STYLES:
            avg_fitness = self.session.query(func.avg(GridCell.current_fitness))\
                .filter(GridCell.style == style)\
                .scalar() or 0.0
            style_fitness[style] = avg_fitness
            
        for category in ATTACK_CATEGORIES:
            avg_fitness = self.session.query(func.avg(GridCell.current_fitness))\
                .filter(GridCell.category == category)\
                .scalar() or 0.0
            category_fitness[category] = avg_fitness
        
        def sample_with_fitness(fitness_dict):
            values = np.array(list(fitness_dict.values()))
            if values.max() > 0:
                weights = 1 - (values / values.max())
            else:
                weights = np.ones_like(values)
            weights = np.exp(weights / self.temperature)
            probs = weights / weights.sum()
            return np.random.choice(list(fitness_dict.keys()), p=probs)
        
        return {
            'style': sample_with_fitness(style_fitness),
            'category': sample_with_fitness(category_fitness)
        }

    def get_grid_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the archive state."""
        total_cells = len(ATTACK_STYLES) * len(ATTACK_CATEGORIES)
        
        filled_cells = self.session.query(GridCell)\
            .filter(GridCell.current_prompt_id.isnot(None))\
            .count()
            
        avg_fitness = self.session.query(func.avg(GridCell.current_fitness))\
            .filter(GridCell.current_prompt_id.isnot(None))\
            .scalar() or 0.0
            
        style_dist = {style: 0 for style in ATTACK_STYLES}
        category_dist = {category: 0 for category in ATTACK_CATEGORIES}
        
        cells = self.session.query(GridCell)\
            .filter(GridCell.current_prompt_id.isnot(None))\
            .all()
        
        for cell in cells:
            style_dist[cell.style] += 1
            category_dist[cell.category] += 1
        
        style_dist = {k: v/total_cells for k, v in style_dist.items()}
        category_dist = {k: v/total_cells for k, v in category_dist.items()}
            
        return {
            'total_cells': total_cells,
            'filled_cells': filled_cells,
            'coverage': filled_cells / total_cells if total_cells > 0 else 0.0,
            'avg_fitness': avg_fitness,
            'style_distribution': style_dist,
            'category_distribution': category_dist
        }

    def save_checkpoint(self):
        """Create a checkpoint of the current archive state."""
        self.session.commit()

    def __del__(self):
        """Cleanup database connection."""
        if hasattr(self, 'session'):
            self.session.close()

    def get_prompt_by_id(self, prompt_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a prompt entry by its ID."""
        try:
            prompt = self.session.query(Prompt).filter_by(id=prompt_id).first()
            if prompt:
                
                return {
                    'id': prompt.id,
                    'prompt': prompt.prompt,
                    'attack_style': prompt.attack_style,
                    'attack_category': prompt.attack_category,
                    'llama_guard_score': prompt.llama_guard_score,
                    'fitness_score': prompt.fitness_score,
                    'target_response': prompt.target_response,
                    'metadata': prompt.extra_data,  
                    'parent_id': prompt.parent_id,
                    'mutation_path': prompt.mutation_path
                }
            return None
        except Exception as e:
            logging.error(f"Error retrieving prompt by ID {prompt_id}: {str(e)}")
            return None
        
    def get_grid_cell(self, style: str, category: str) -> Optional[GridCell]:
        """Get the grid cell for a given style and category."""
        return self.session.query(GridCell).filter_by(
            style=style,
            category=category
    ).first()

    def _evaluate_effectiveness(self, 
                              fitness_score: float, 
                              parent_fitness: Optional[float],
                              component_scores: Dict[str, float]) -> Tuple[bool, str]:
        """Evaluate prompt effectiveness based on multiple criteria."""
        if parent_fitness is None:
            return True, "Initial prompt"
            
        
        improvement_threshold = 0.05  
        
        if fitness_score > parent_fitness * (1 + improvement_threshold):
            return True, f"Improved fitness by {((fitness_score/parent_fitness) - 1) * 100:.2f}%"
            
        
        if any(score > 0.8 for score in component_scores.values()):
            return True, "Outstanding component scores"
            
        return False, f"Insufficient improvement over parent (parent: {parent_fitness:.3f}, current: {fitness_score:.3f})"

    def _update_grid_cell(self, prompt: Prompt):
        """Update grid cell with new effective prompt."""
        cell = self.session.query(GridCell).filter_by(
            style=prompt.attack_style,
            category=prompt.attack_category
        ).first()
        
        if not cell:
            raise ValueError(f"No grid cell found for style={prompt.attack_style}, category={prompt.attack_category}")
        
        
        cell.current_prompt_id = prompt.id
        cell.current_fitness = prompt.fitness_score
        cell.update_count += 1
        cell.last_update = datetime.now(timezone.utc)

    
    def get_prompt_history(self, prompt_id: int) -> List[Dict[str, Any]]:
        """Get the complete mutation history of a prompt."""
        history = []
        current_id = prompt_id
        
        while current_id:
            prompt = self.session.query(Prompt).get(current_id)
            if not prompt:
                break
                
            history.append({
                'id': prompt.id,
                'prompt': prompt.prompt,
                'fitness_score': prompt.fitness_score,
                'is_effective': prompt.is_effective,
                'effectiveness_reason': prompt.effectiveness_reason,
                'generation': prompt.generation,
                'timestamp': prompt.timestamp
            })
            
            current_id = prompt.parent_id
            
        return history
    
    def get_all_prompts(self, 
                    effective_only: bool = False, 
                    min_fitness: Optional[float] = None,
                    style: Optional[str] = None,
                    category: Optional[str] = None,
                    limit: Optional[int] = None,
                    mutation_type: Optional[str] = None,
                    max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all prompts with flexible filtering.
        
        Args:
            effective_only: Filter for effective prompts only
            min_fitness: Minimum fitness score filter
            style: Filter by attack style
            category: Filter by attack category
            limit: Maximum number of prompts to return
            mutation_type: Filter by mutation type
            max_length: Maximum prompt length filter
        """
        query = self.session.query(Prompt).order_by(Prompt.timestamp.desc())
        
        if effective_only:
            query = query.filter(Prompt.is_effective == True)
        if min_fitness is not None:
            query = query.filter(Prompt.fitness_score >= min_fitness)
        if style:
            query = query.filter(Prompt.attack_style == style)
        if category:
            query = query.filter(Prompt.attack_category == category)
        if mutation_type:
            query = query.filter(Prompt.mutation_type == mutation_type)
        if max_length:
            query = query.filter(func.length(Prompt.prompt) <= max_length)
                
        
        if limit:
            query = query.limit(limit)
                
        prompts = query.all()
        
        return [{
            'id': p.id,
            'prompt': p.prompt,
            'attack_style': p.attack_style,
            'attack_category': p.attack_category,
            'fitness_score': p.fitness_score,
            'is_effective': p.is_effective,
            'effectiveness_score': p.effectiveness_score,
            'generation': p.generation,
            'timestamp': p.timestamp,
            'mutation_type': p.mutation_type,  
            'target_response': p.target_response,  
            'extra_data': p.extra_data,  
            'parent_id': p.parent_id,  
            'mutation_path': p.mutation_path  
        } for p in prompts]
    
    def update_prompt_response(self, prompt_id: int, target_response: str) -> bool:
        """Update target response for prompts with deferred generation."""
        try:
            with self.transaction():
                prompt = self.session.query(Prompt).get(prompt_id)
                if not prompt:
                    return False
                    
                prompt.target_response = target_response
                prompt.extra_data = {
                    **(prompt.extra_data or {}),
                    'needs_response': False,
                    'response_added_at': datetime.now().isoformat()
                }
                
                
                if prompt.is_effective:
                    self._update_grid_cell(prompt)
                    
                return True
                
        except Exception as e:
            logging.error(f"Error updating prompt response: {str(e)}")
            return False
        
    
    def get_prompts_needing_response(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get prompts that need target responses."""
        query = self.session.query(Prompt).filter(
            Prompt.mutation_type == MUTATION_TYPES['SUB_MUTATOR'],
            Prompt.target_response.is_(None)
        )
        if limit:
            query = query.limit(limit)
        return [{
            'id': p.id,
            'prompt': p.prompt,
            'attack_style': p.attack_style,
            'attack_category': p.attack_category
        } for p in query.all()]