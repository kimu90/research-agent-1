import logging
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)
import os
from datetime import datetime
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from research_agent.db.db import ContentDB
from tools.research.common.model_schemas import ContentItem

logger = logging.getLogger(__name__)

class DatabaseManager:
    _instance = None

    def __init__(self):
        self.db = None
        self.initialize_db()

    @classmethod
    def get_instance(cls) -> 'DatabaseManager':
        if cls._instance is None:
            cls._instance = DatabaseManager()
        return cls._instance

    def initialize_db(self):
        try:
            self.db = ContentDB(os.environ.get('DB_PATH', '/data/content.db'))
            logger.info("Database connection initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            self.db = None

    def store_content(self, content_item: ContentItem) -> bool:
        """Store or update content in database"""
        try:
            if self.db:
                is_new = self.db.upsert_doc(content_item)
                action = "stored" if is_new else "updated"
                logger.info(f"Content {action} successfully (ID: {getattr(content_item, 'id', 'N/A')})")
                return is_new
            return False
        except Exception as e:
            logger.error(f"Error storing content: {str(e)}")
            return False

    def store_evaluation(self, eval_type: str, data: Dict[str, Any]) -> bool:
        """Store evaluation results"""
        try:
            if not self.db:
                return False

            store_methods = {
                'accuracy': self.db.store_accuracy_evaluation,
                'coverage': self.db.store_source_coverage,
                'coherence': self.db.store_logical_coherence,
                'relevance': self.db.store_answer_relevance
            }

            if eval_type in store_methods:
                store_methods[eval_type]({
                    'timestamp': datetime.now().isoformat(),
                    **data
                })
                logger.info(f"Stored {eval_type} evaluation successfully")
                return True
            else:
                logger.error(f"Unknown evaluation type: {eval_type}")
                return False

        except Exception as e:
            logger.error(f"Error storing {eval_type} evaluation: {str(e)}")
            return False

    def get_evaluations(self, eval_type: str, limit: int = 50) -> list:
        """Retrieve evaluation results"""
        try:
            if not self.db:
                return []

            get_methods = {
                'accuracy': self.db.get_accuracy_evaluations,
                'coverage': self.db.get_source_coverage_evaluations,
                'coherence': self.db.get_logical_coherence_evaluations,
                'relevance': self.db.get_answer_relevance_evaluations
            }

            if eval_type in get_methods:
                return get_methods[eval_type](limit=limit)
            else:
                logger.error(f"Unknown evaluation type: {eval_type}")
                return []

        except Exception as e:
            logger.error(f"Error retrieving {eval_type} evaluations: {str(e)}")
            return []

# Singleton instance getter
def get_db() -> Optional[DatabaseManager]:
    return DatabaseManager.get_instance()