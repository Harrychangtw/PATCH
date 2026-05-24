
from pathlib import Path
import os


PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
LOG_DIR = DATA_DIR / "logs"
SEED_PROMPTS_DIR = DATA_DIR / "seed_prompts"

MODEL_ARCHIVE_DIR = Path(os.getenv('MODEL_ARCHIVE_DIR', PROJECT_ROOT / "_model_archive"))


DB_NAME = "Qwen2.5-7B-Instruct-Q8_0.gguf.db"

DB_PATH = Path(os.getenv('DB_PATH', str(DATA_DIR / DB_NAME)))


DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DB_MONITOR_THRESHOLD = 500


MAX_ITERATIONS = int(os.getenv('MAX_ITERATIONS', '4000'))
SIMILARITY_THRESHOLD = 0.6
SAMPLING_TEMPERATURE = 0.7
LOW_FITNESS_BIAS = 0.7
CHECKPOINT_FREQUENCY = 100


MIN_LLAMAGUARD_SCORE = 0.5


VRAM_LIMIT_GB = float(os.getenv('VRAM_LIMIT_GB', '23.9'))
SYSTEM_RAM_LIMIT_GB = float(os.getenv('SYSTEM_RAM_LIMIT_GB', '60'))
PRIMARY_DEVICE = os.getenv('PRIMARY_DEVICE', 'cuda:0')

for directory in [DATA_DIR, CHECKPOINT_DIR, LOG_DIR, SEED_PROMPTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)