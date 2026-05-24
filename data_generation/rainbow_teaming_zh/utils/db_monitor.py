import os
import sqlite3
from datetime import datetime
import subprocess
from typing import Optional

class DBMonitor:
    def __init__(self, db_path: str, threshold: int = 100):
        """
        Initialize the database monitor.
        
        Args:
            db_path (str): Path to the SQLite database
            threshold (int): Number of new entries that trigger a commit
        """
        self.db_path = db_path
        self.threshold = threshold
        self.previous_count = self._get_current_count()
        
        print(f"Initialized DBMonitor with database: {db_path}")
        print(f"Initial prompt count: {self.previous_count}")

    def _get_current_count(self) -> int:
        """Get the current count of entries in the prompts table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM prompts")
                return cursor.fetchone()[0]
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return 0

    def _push_to_remote(self) -> bool:
        """Push the committed changes to the remote repository."""
        try:
            
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            current_branch = result.stdout.strip()
            
            
            subprocess.run(
                ['git', 'push', 'origin', current_branch],
                check=True
            )
            
            print(f"Successfully pushed changes to remote branch: {current_branch}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Git push error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during git push: {e}")
            return False

    def _commit_to_git(self) -> bool:
        """Commit the database changes to git and push to remote."""
        try:
            timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            current_count = self._get_current_count()
            commit_message = f"database snapshot {timestamp} (prompts: {current_count})"
            
            
            subprocess.run(['git', 'add', self.db_path], check=True)
            
            
            subprocess.run(['git', 'commit', '-m', commit_message], check=True)
            
            print(f"Successfully committed changes: {commit_message}")
            
            
            if self._push_to_remote():
                return True
            else:
                print("Warning: Commit successful but push failed")
                return False
            
        except subprocess.CalledProcessError as e:
            print(f"Git command error: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during git operations: {e}")
            return False

    def check_and_commit(self) -> Optional[bool]:
        """
        Check if database has grown beyond threshold and commit if necessary.
        
        Returns:
            bool: True if commit was made, False if no commit needed or error occurred
            None: If there was an error getting the count
        """
        try:
            current_count = self._get_current_count()
            
            if current_count - self.previous_count >= self.threshold:
                print(f"Threshold reached: Previous={self.previous_count}, Current={current_count}")
                
                
                if self._commit_to_git():
                    self.previous_count = current_count
                    return True
                return False
            
            print(f"Below threshold: Previous={self.previous_count}, Current={current_count}")
            return False
            
        except Exception as e:
            print(f"Error in check_and_commit: {e}")
            return None

def test_monitor(db_path: str, threshold: int = 0):
    """Test function to demonstrate DBMonitor usage"""
    try:
        
        if not os.path.exists(db_path):
            print(f"Error: Database not found at {db_path}")
            return
            
        
        print("\nInitializing DB Monitor...")
        monitor = DBMonitor(db_path, threshold)
        
        
        print("\nChecking database changes...")
        result = monitor.check_and_commit()
        
        if result is True:
            print("Success: Changes committed and pushed to git")
        elif result is False:
            print("Info: No commit needed")
        else:
            print("Error: Failed to check database")
            
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":

    DB_PATH = r"your_database_path_here"    
    test_monitor(DB_PATH)