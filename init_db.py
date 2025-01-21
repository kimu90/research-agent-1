
import os
import logging
from research_agent.db.db import ContentDB
from tools.research.common.model_schemas import ContentItem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_database(db_path):
    """
    Initialize the database with proper error handling.
    """
    try:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if not os.path.exists(db_path) or os.path.getsize(db_path) == 0:
            logger.info(f"Initializing database at {db_path}")
            db = ContentDB(db_path)
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
        else:
            logger.info("Database already exists and is non-empty; skipping initialization")
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise

if __name__ == "__main__":
    db_path = os.getenv('DB_PATH', '/data/content.db')
    init_db = os.getenv('INIT_DB', 'false').lower() == 'true'

    if init_db:
        logger.info("INIT_DB is set to true. Running database initialization.")
        initialize_database(db_path)
    else:
        logger.info("INIT_DB is set to false. Skipping database initialization.")

