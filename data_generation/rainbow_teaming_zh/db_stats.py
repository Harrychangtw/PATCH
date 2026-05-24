from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from core.database import Base, Prompt, GridCell
from config.settings import DB_PATH

class DatabaseStatsOrganizer:
    def __init__(self, db_path=None):
        """Initialize the database connection."""
        db_file = db_path or DB_PATH
        self.engine = create_engine(f'sqlite:///{db_file}', echo=False)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        
        self.attack_categories = [
            '暴力犯罪', '非暴力犯罪', '性相關犯罪',
            '兒童性剝削', '誹謗', '專業建議', '隱私',
            '智慧財產權', '無差別武器', '仇恨',
            '自殺與自殘', '性內容', '選舉'
        ]
    
    def get_table_counts(self):
        """Get the total count of entries in each table."""
        prompt_count = self.session.query(Prompt).count()
        grid_cell_count = self.session.query(GridCell).count()
        
        return {
            'Prompt': prompt_count,
            'GridCell': grid_cell_count
        }
    
    def get_category_breakdown(self):
        """Get counts of prompts by attack category."""
        category_counts = {}
        
        for category in self.attack_categories:
            count = self.session.query(Prompt).filter(
                Prompt.attack_category == category
            ).count()
            category_counts[category] = count
        
        return category_counts
    
    def get_effective_prompt_stats(self):
        """Get statistics about effective vs. non-effective prompts."""
        effective_count = self.session.query(Prompt).filter(
            Prompt.is_effective == True
        ).count()
        
        total_count = self.session.query(Prompt).count()
        non_effective_count = total_count - effective_count
        
        return {
            'effective': effective_count,
            'non_effective': non_effective_count,
            'effective_percentage': (effective_count / total_count * 100) if total_count > 0 else 0
        }
    
    def display_stats(self):
        """Display formatted statistics to the console."""
        table_counts = self.get_table_counts()
        category_breakdown = self.get_category_breakdown()
        effective_stats = self.get_effective_prompt_stats()
        
        print("=" * 50)
        print("DATABASE STATISTICS")
        print("=" * 50)
        
        print("\nTABLE COUNTS:")
        for table, count in table_counts.items():
            print(f"  {table}: {count}")
        
        print("\nCATEGORY BREAKDOWN:")
        for category, count in category_breakdown.items():
            print(f"  {category}: {count}")
        
        print("\nEFFECTIVE PROMPT STATS:")
        print(f"  Effective: {effective_stats['effective']}")
        print(f"  Non-effective: {effective_stats['non_effective']}")
        print(f"  Effective %: {effective_stats['effective_percentage']:.2f}%")
        
        print("=" * 50)


if __name__ == "__main__":
    stats_organizer = DatabaseStatsOrganizer()
    stats_organizer.display_stats()
