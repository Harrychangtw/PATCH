

import nltk
import logging
from pathlib import Path

def setup_nltk():
    """Download required NLTK data packages."""
    required_packages = [
        'punkt',            
        'punkt_tab',        
        'averaged_perceptron_tagger',
        'wordnet'
    ]
    
    logging.info("Setting up NLTK packages...")
    
    try:
        
        nltk.download('punkt', quiet=True)
        
        
        try:
            
            from nltk.tokenize import word_tokenize
            word_tokenize("Test sentence")
        except LookupError:
            
            nltk.download('punkt_tab', quiet=True)
            
        
        for package in required_packages:
            if package not in ['punkt', 'punkt_tab']:  
                nltk.download(package, quiet=True)
                logging.info(f"Successfully downloaded NLTK package: {package}")
                
    except Exception as e:
        logging.error(f"Error during NLTK setup: {str(e)}")
        raise

    logging.info("NLTK setup completed successfully")

if __name__ == "__main__":
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    setup_nltk()