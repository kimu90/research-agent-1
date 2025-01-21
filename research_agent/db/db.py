from tools.research.common.model_schemas import ContentItem
from typing import Optional, Dict, Any
import threading
import logging
import sqlite3
import os
import json
from datetime import datetime

class ContentDB:
    def __init__(self, db_path: str = ":memory:"):
        """
        Initializes the ContentDB instance, setting up an SQLite database.

        Args:
            db_path (str): The file path to the SQLite database. Defaults to an in-memory database.
                           This allows for persistent data storage when a file path is provided.

        This constructor ensures the database contains 'content' and 'factual_accuracy' tables.
        """
        self.lock = threading.Lock()  # Ensures that database operations are thread-safe

        if db_path != ":memory:":
            # Ensures the directory for the database file exists
            db_dir = os.path.dirname(db_path)
            if not os.path.exists(db_dir):
                os.makedirs(db_dir)

        # Allow multi-threaded access to the database by setting check_same_thread to False
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        
        with self.lock:
            # Create content table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS content (
                    id TEXT PRIMARY KEY,
                    url TEXT UNIQUE,
                    title TEXT,
                    snippet TEXT,
                    content TEXT,
                    source TEXT
                )
                """
            )
            
            # Create factual accuracy evaluation table
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS factual_accuracy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    timestamp TEXT,
                    factual_score REAL,
                    total_sources INTEGER,
                    citation_accuracy REAL,
                    claim_details TEXT
                )
                """
            )
            
            self.conn.commit()

    def get_doc_by_id(self, id: str) -> Optional[ContentItem]:
        """
        Retrieves a document by its unique ID.

        Args:
            id (str): The unique identifier for the document.

        Returns:
            Optional[ContentItem]: A ContentItem instance if found, else None.
        """
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT id, url, title, snippet, content, source FROM content WHERE id = ?",
                (id,),
            )
            row = cursor.fetchone()
            return (
                ContentItem(
                    **dict(
                        zip(["id", "url", "title", "snippet", "content", "source"], row)
                    )
                )
                if row
                else None
            )

    def get_doc_by_url(self, url: str) -> Optional[ContentItem]:
        """
        Retrieves a document by its URL.

        Args:
            url (str): The URL associated with the document.

        Returns:
            Optional[ContentItem]: A ContentItem instance if found, else None.
        """
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT id, url, title, snippet, content, source FROM content WHERE url = ?",
                (url,),
            )
            row = cursor.fetchone()
            return (
                ContentItem(
                    **dict(
                        zip(["id", "url", "title", "snippet", "content", "source"], row)
                    )
                )
                if row
                else None
            )

    def upsert_doc(self, doc: ContentItem) -> bool:
        """
        Inserts a new document or updates an existing one based on the URL conflict.

        Args:
            doc (ContentItem): A ContentItem instance containing the document data.

        Returns:
            bool: True if the document is new, False if updated
        """
        with self.lock:
            cursor = self.conn.cursor()
            try:
                # Check if document already exists
                cursor.execute("SELECT 1 FROM content WHERE url = ?", (doc.url,))
                is_new = cursor.fetchone() is None

                cursor.execute(
                    """
                    INSERT INTO content (id, url, title, snippet, content, source)
                    VALUES (:id, :url, :title, :snippet, :content, :source)
                    ON CONFLICT(url) DO UPDATE SET
                    id=excluded.id,
                    title=excluded.title,
                    snippet=excluded.snippet,
                    content=excluded.content,
                    source=excluded.source
                    """,
                    doc.to_dict(),
                )
                self.conn.commit()
                logging.info(f"Document {'inserted' if is_new else 'updated'} successfully: {doc.id}")
                return is_new
            except sqlite3.IntegrityError as e:
                logging.error(f"Error inserting/updating document: {e}")
                raise

    def store_accuracy_evaluation(self, accuracy_data: Dict[str, Any]):
        """
        Store factual accuracy evaluation results
        
        Args:
            accuracy_data (dict): Dictionary containing accuracy evaluation details
        """
        with self.lock:
            try:
                # Prepare data for storage
                insert_data = {
                    'query': accuracy_data.get('query', 'Unknown'),
                    'timestamp': accuracy_data.get('timestamp', datetime.now().isoformat()),
                    'factual_score': accuracy_data.get('factual_score', 0.0),
                    'total_sources': accuracy_data.get('total_sources', 0),
                    'citation_accuracy': accuracy_data.get('citation_accuracy', 0.0),
                    'claim_details': json.dumps(accuracy_data.get('claim_details', []))
                }

                # Insert accuracy data
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO factual_accuracy 
                    (query, timestamp, factual_score, total_sources, citation_accuracy, claim_details)
                    VALUES (:query, :timestamp, :factual_score, :total_sources, :citation_accuracy, :claim_details)
                    """,
                    insert_data
                )
                
                self.conn.commit()
                logging.info("Factual accuracy evaluation stored successfully")
                return cursor.lastrowid
            except Exception as e:
                logging.error(f"Error storing factual accuracy: {str(e)}")
                self.conn.rollback()
                return None

    def get_accuracy_evaluations(self, query: Optional[str] = None, limit: int = 10):
        """
        Retrieve factual accuracy evaluations
        
        Args:
            query (Optional[str]): Optional query to filter results
            limit (int): Maximum number of results to return
        
        Returns:
            List of accuracy evaluation dictionaries
        """
        with self.lock:
            cursor = self.conn.cursor()
            try:
                if query:
                    cursor.execute(
                        """
                        SELECT id, query, timestamp, factual_score, total_sources, 
                               citation_accuracy, claim_details 
                        FROM factual_accuracy 
                        WHERE query LIKE ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                        """, 
                        (f'%{query}%', limit)
                    )
                else:
                    cursor.execute(
                        """
                        SELECT id, query, timestamp, factual_score, total_sources, 
                               citation_accuracy, claim_details 
                        FROM factual_accuracy 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                        """, 
                        (limit,)
                    )
                
                # Fetch and process results
                columns = ['id', 'query', 'timestamp', 'factual_score', 'total_sources', 'citation_accuracy', 'claim_details']
                results = []
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    # Parse JSON claim details
                    result['claim_details'] = json.loads(result['claim_details']) if result['claim_details'] else []
                    results.append(result)
                
                return results
            except Exception as e:
                logging.error(f"Error retrieving accuracy evaluations: {str(e)}")
                return []

    def delete_doc(self, id: str):
        """
        Deletes a document by its ID.

        Args:
            id (str): The unique identifier for the document to be deleted.
        """
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM content WHERE id = ?", (id,))
            self.conn.commit()

    def generate_snippet(self, text: str) -> str:
        """
        Generates a text snippet from the provided text.

        Args:
            text (str): The text from which to generate the snippet.

        Returns:
            str: A string representing the snippet, truncated to 150 characters plus an ellipsis.
        """
        return text[:150] + "..."  # Simplistic snippet generation for demo purposes

    def close(self):
        """
        Closes the database connection
        """
        with self.lock:
            self.conn.close()