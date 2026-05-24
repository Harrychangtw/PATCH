
from pathlib import Path
import os

def setup_project_structure():
    """Create necessary directories and __init__.py files."""
    
    project_root = Path(__file__).parent.parent
    
    
    directories = [
        '',  
        'config',
        'core',
        'models',
        'scripts',
        'utils'
    ]
    
    
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        init_file = dir_path / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            print(f"Created {init_file}")
    
    
    data_dirs = [
        'data',
        'data/checkpoints',
        'data/logs',
        'data/seed_prompts'
    ]
    
    for dir_name in data_dirs:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"Created directory: {dir_path}")

if __name__ == "__main__":
    setup_project_structure()