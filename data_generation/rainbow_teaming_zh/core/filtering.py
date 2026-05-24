
from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

class SimilarityFilter:
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.smoother = SmoothingFunction()

    def is_similar(self, prompt1: str, prompt2: str) -> float:
        """Check if two prompts are too similar using smoothed BLEU score."""
        tokens1 = word_tokenize(prompt1.lower())
        tokens2 = word_tokenize(prompt2.lower())
        
        
        bleu_score = sentence_bleu(
            [tokens1], 
            tokens2, 
            smoothing_function=self.smoother.method7,
            weights=(0.25, 0.25, 0.25, 0.25)  
        )
        
        return bleu_score > self.threshold