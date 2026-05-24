
from typing import List, Tuple
from nltk.tokenize import word_tokenize
import numpy as np

class MetricsCalculator:
    @staticmethod
    def calculate_bleu(reference: str, candidate: str) -> float:
        """Calculate similarity between two texts using a simplified approach."""
        try:
            
            reference_words = set(word_tokenize(reference.lower()))
            candidate_words = set(word_tokenize(candidate.lower()))
            
            if not reference_words or not candidate_words:
                return 0.0
                
            overlap = len(reference_words.intersection(candidate_words))
            similarity = overlap / max(len(reference_words), len(candidate_words))
            
            return similarity
            
        except Exception as e:
            print(f"Error in similarity calculation: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_response_similarity(response1: str, response2: str) -> float:
        """Calculate similarity between two responses."""
        return MetricsCalculator.calculate_bleu(response1, response2)

    @staticmethod
    def calculate_prompt_effectiveness(safety_score: float, response_diversity: float) -> float:
        """Calculate overall effectiveness of a prompt."""
        
        return 0.7 * safety_score + 0.3 * response_diversity