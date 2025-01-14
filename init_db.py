import os
from research_agent.db.db import ContentDB
from tools.research.common.model_schemas import ContentItem
import logging

def initialize_database():
    """
    Initialize the database with proper error handling and logging.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    db_path = os.getenv('DB_PATH', '/data/content.db')
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        logger.info(f"Initializing database at {db_path}")
        db = ContentDB(db_path)
        
        # Add sample document only if database is empty
        if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
            sample_doc = ContentItem(
                id="sample1",
                url="https://example.com",
                title="Research Agent Database",
                snippet="Initialized research database for content storage",
                content="This is an initialized database for the Research Agent system.",
                source="system_init"
            )
            
            db.upsert_doc(sample_doc)
            logger.info("Sample document added to database")
        
        logger.info("Database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during database initialization: {str(e)}")
        raise

if __name__ == "__main__":
    initialize_database()