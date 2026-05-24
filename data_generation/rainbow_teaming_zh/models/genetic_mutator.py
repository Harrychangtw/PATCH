import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json
from datetime import datetime
import os
import sqlite3
import logging
from pathlib import Path


from config.constants import (
    ATTACK_STYLES, 
    ATTACK_CATEGORIES,
    ATTACK_CATEGORY_DEFINITIONS,
    STYLE_TECHNIQUES
)

@dataclass
class MutationConfig:
    population_size: int = 30
    generations: int = 20
    mutation_rate: float = 0.4
    crossover_rate: float = 0.7
    tournament_size: int = 3
    elite_size: int = 5
    min_prompt_length: int = 20
    max_prompt_length: int = 200

@dataclass
class Prompt:
    text: str
    style: str
    category: str
    fitness: float = 0.0
    ancestry: List[str] = None
    llama_guard_score: float = 0.0
    safety_score: float = 0.0
    
    def __post_init__(self):
        if self.ancestry is None:
            self.ancestry = []

class ChineseGeneticMutationTester:
    def __init__(self, db_path: str, config: MutationConfig = None):
        """Initialize with database path and configuration."""
        self.db_path = db_path
        self.config = config or MutationConfig()
        self.population: List[Prompt] = []
        self.generation = 0
        self.best_prompts: List[Prompt] = []
        
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _load_seed_prompts(self) -> List[Prompt]:
        """Load existing prompts from database as seeds."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            
            cursor.execute("""
                SELECT prompt, attack_style, attack_category, llama_guard_score, 
                       safety_score, fitness_score
                FROM prompts 
                WHERE fitness_score > 0.7
                ORDER BY fitness_score DESC
                LIMIT 50
            """)
            
            results = cursor.fetchall()
            prompts = []
            
            for row in results:
                prompts.append(Prompt(
                    text=row[0],
                    style=row[1],
                    category=row[2],
                    llama_guard_score=row[3],
                    safety_score=row[4],
                    fitness=row[5],
                    ancestry=["Seed from database"]
                ))
            
            logging.info(f"Loaded {len(prompts)} seed prompts from database")
            return prompts
            
        except Exception as e:
            logging.error(f"Error loading seed prompts: {str(e)}")
            return []
        finally:
            conn.close()

    def _create_variation(self, text: str, style: str, category: str) -> str:
        """Create a Chinese variation of a prompt using style and category modifiers."""
        techniques = STYLE_TECHNIQUES.get(style, {})
        style_keywords = techniques.get("關鍵字", [])
        patterns = techniques.get("語言模式", [])
        
        
        category_def = ATTACK_CATEGORY_DEFINITIONS.get(category, "")
        category_keywords = category_def.split("、") if category_def else []
        
        
        words = text.split()
        if len(words) > 3:
            
            if style_keywords and random.random() < 0.7:
                words.insert(
                    random.randint(0, len(words)), 
                    random.choice(style_keywords)
                )
            
            
            if patterns and random.random() < 0.6:
                pattern = random.choice(patterns).replace("[", "").replace("]", "")
                words.insert(0, pattern)
            
            
            if category_keywords and random.random() < 0.5:
                words.append(random.choice(category_keywords))
                
        return "".join(words)

    def _crossover(self, parent1: Prompt, parent2: Prompt) -> Tuple[Prompt, Prompt]:
        """Perform crossover between two Chinese prompts."""
        if random.random() > self.config.crossover_rate:
            return parent1, parent2
        
        
        segments1 = parent1.text.split("，")
        segments2 = parent2.text.split("，")
        
        if len(segments1) < 2 or len(segments2) < 2:
            return parent1, parent2
            
        
        point = random.randint(1, min(len(segments1), len(segments2)) - 1)
        
        
        child1_text = "，".join(segments1[:point] + segments2[point:])
        child2_text = "，".join(segments2[:point] + segments1[point:])
        
        return (
            Prompt(
                text=child1_text,
                style=random.choice([parent1.style, parent2.style]),
                category=random.choice([parent1.category, parent2.category]),
                ancestry=parent1.ancestry + ["Crossover with " + parent2.text[:30]]
            ),
            Prompt(
                text=child2_text,
                style=random.choice([parent1.style, parent2.style]),
                category=random.choice([parent1.category, parent2.category]),
                ancestry=parent2.ancestry + ["Crossover with " + parent1.text[:30]]
            )
        )

    def _mutate(self, prompt: Prompt) -> Prompt:
        """Mutate a Chinese prompt."""
        if random.random() > self.config.mutation_rate:
            return prompt
        
        text = prompt.text
        techniques = STYLE_TECHNIQUES.get(prompt.style, {})
        
        mutation_ops = [
            
            lambda t: t + random.choice(techniques.get("關鍵字", [""])),
            
            
            lambda t: random.choice(techniques.get("語言模式", [""])) + t,
            
            
            lambda t: t + "，" + random.choice(
                ATTACK_CATEGORY_DEFINITIONS.get(prompt.category, "").split("、")[:5]
            )
        ]
        
        
        new_text = random.choice(mutation_ops)(text)
        
        return Prompt(
            text=new_text,
            style=prompt.style,
            category=prompt.category,
            ancestry=prompt.ancestry + ["Mutation"]
        )

    def _calculate_fitness(self, prompt: Prompt) -> float:
        """Calculate fitness for Chinese prompts."""
        
        length = len(prompt.text)
        length_score = min(length / 100, 1.0) if length < 200 else (200 / length)
        
        
        techniques = STYLE_TECHNIQUES.get(prompt.style, {})
        style_keywords = techniques.get("關鍵字", [])
        style_score = sum(1 for keyword in style_keywords if keyword in prompt.text) / len(style_keywords) if style_keywords else 0.5
        
        
        category_keywords = ATTACK_CATEGORY_DEFINITIONS.get(prompt.category, "").split("、")
        category_score = sum(1 for keyword in category_keywords if keyword in prompt.text) / len(category_keywords) if category_keywords else 0.5
        
        
        fitness = (
            0.3 * length_score +
            0.35 * style_score +
            0.35 * category_score
        )
        
        return min(max(fitness, 0.0), 1.0)

    def _select_parent(self, population: List[Prompt]) -> Prompt:
        """Select parent using tournament selection."""
        tournament = random.sample(population, self.config.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def evolve(self, output_file: str = None):
        """Run genetic evolution process."""
        
        seed_prompts = self._load_seed_prompts()
        if not seed_prompts:
            logging.error("No seed prompts available from database")
            return
        
        
        self.population = seed_prompts
        while len(self.population) < self.config.population_size:
            seed = random.choice(seed_prompts)
            variation = self._create_variation(
                seed.text, 
                random.choice(ATTACK_STYLES),
                random.choice(ATTACK_CATEGORIES)
            )
            self.population.append(Prompt(
                text=variation,
                style=random.choice(ATTACK_STYLES),
                category=random.choice(ATTACK_CATEGORIES),
                ancestry=["Variation of " + seed.text[:30]]
            ))
        
        
        for generation in range(self.config.generations):
            logging.info(f"\nGeneration {generation + 1}/{self.config.generations}")
            
            
            for prompt in self.population:
                prompt.fitness = self._calculate_fitness(prompt)
            
            
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            
            self.best_prompts = self.population[:self.config.elite_size]
            
            
            new_population = self.population[:self.config.elite_size]
            
            
            while len(new_population) < self.config.population_size:
                parent1 = self._select_parent(self.population)
                parent2 = self._select_parent(self.population)
                
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.config.population_size]
            
            
            best_fitness = max(p.fitness for p in self.population)
            avg_fitness = sum(p.fitness for p in self.population) / len(self.population)
            logging.info(f"Best Fitness: {best_fitness:.3f}, Average Fitness: {avg_fitness:.3f}")
            
            
            best_prompt = max(self.population, key=lambda x: x.fitness)
            logging.info(f"Best Prompt: {best_prompt.text[:50]}...")
            
        
        if output_file:
            self._save_results(output_file)

    def _save_results(self, output_file: str):
        """Save evolution results to file."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "best_prompts": [
                {
                    "text": prompt.text,
                    "style": prompt.style,
                    "category": prompt.category,
                    "fitness": prompt.fitness,
                    "ancestry": prompt.ancestry
                }
                for prompt in self.best_prompts
            ]
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Results saved to {output_file}")

def test_genetic_mutation():
    
    db_path = "path/to/your/database.db"  
    
    
    config = MutationConfig(
        population_size=30,
        generations=20,
        mutation_rate=0.4,
        crossover_rate=0.7,
        tournament_size=3,
        elite_size=5
    )
    
    
    tester = ChineseGeneticMutationTester(db_path, config)
    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"mutation_results/chinese_mutation_results_{timestamp}.json"
    
    
    print("Starting Chinese prompt evolution...")
    tester.evolve(output_file)
    
    
    print("\nTop 5 Evolved Prompts:")
    for i, prompt in enumerate(tester.best_prompts[:5], 1):
        print(f"\n{i}. 提示: {prompt.text}")
        print(f"   風格: {prompt.style}")
        print(f"   類別: {prompt.category}")
        print(f"   適應度: {prompt.fitness:.3f}")
        print(f"   演化歷程: {' -> '.join(prompt.ancestry)}")


if __name__ == "__main__":
    test_genetic_mutation()