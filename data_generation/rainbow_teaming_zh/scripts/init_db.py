import sys
import random
from pathlib import Path


project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.db_initializer import DatabaseInitializer

if __name__ == "__main__":
    random.seed(42)  
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else None
    initializer = DatabaseInitializer(db_path=db_path)
    initializer.init_database()
    initializer.init_with_seeds()
    initializer.cleanup()