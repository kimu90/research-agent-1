import os
import logging
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)
from .db import ContentDB

_db_instance = None

def get_db_connection():
    global _db_instance
    if _db_instance is None:
        try:
            _db_instance = ContentDB(os.environ.get('DB_PATH', '/data/content.db'))
            logging.info("Database connection initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing database: {str(e)}")
            _db_instance = None
    return _db_instance