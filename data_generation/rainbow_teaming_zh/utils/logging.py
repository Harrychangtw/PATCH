
import logging
import sys
from pathlib import Path
import colorlog
import codecs
import locale

def setup_logging(log_dir: Path, log_level: int = logging.DEBUG) -> logging.Logger:
    """Setup logging with proper Unicode handling and colorized output."""
    
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    
    if sys.platform == 'win32':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleCP(65001)
            kernel32.SetConsoleOutputCP(65001)
        except:
            pass

    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(levelname)s - %(message)s%(reset)s",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'SUCCESS': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        },
        reset=True
    )

    
    class CharacterFilter(logging.Filter):
        def filter(self, record):
            try:
                if isinstance(record.msg, bytes):
                    record.msg = record.msg.decode('utf-8', errors='replace')
                elif isinstance(record.msg, str):
                    record.msg = record.msg.encode('utf-8', errors='replace').decode('utf-8')
                return True
            except:
                return False

    
    file_handler = logging.FileHandler(
        log_dir / 'rainbow_teaming.log',
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)

    
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    
    char_filter = CharacterFilter()
    file_handler.addFilter(char_filter)
    console_handler.addFilter(char_filter)

    
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    
    root_logger.debug(f"Console encoding: {sys.stdout.encoding}")
    root_logger.debug(f"File system encoding: {sys.getfilesystemencoding()}")
    root_logger.debug(f"Locale encoding: {locale.getpreferredencoding()}")

    return root_logger

def log_success(message: str):
    """Log success messages with proper encoding."""
    try:
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        elif isinstance(message, str):
            message = message.encode('utf-8', errors='replace').decode('utf-8')
        logging.info(f"\033[92m[SUCCESS] {message}\033[0m")
    except Exception as e:
        logging.error(f"Error logging success message: {str(e)}")

def log_chinese(message: str, level: int = logging.INFO):
    """Safely log messages containing Chinese characters."""
    try:
        if isinstance(message, bytes):
            message = message.decode('utf-8', errors='replace')
        elif isinstance(message, str):
            message = message.encode('utf-8', errors='replace').decode('utf-8')
        logging.log(level, message)
    except Exception as e:
        logging.error(f"Error logging Chinese message: {str(e)}")

class SafeLogger:
    """Context manager for safely logging Chinese text."""
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger()
        self._original_handlers = []

    def __enter__(self):
        
        self._original_handlers = self.logger.handlers[:]
        
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.encoding = 'utf-8'
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        
        self.logger.handlers = self._original_handlers