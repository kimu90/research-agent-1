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

            # Source coverage evaluation table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS source_coverage_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    coverage_score REAL,
                    coverage_ratio REAL,
                    diversity_score REAL,
                    missed_sources TEXT,
                    total_sources INTEGER
                )
            """)

            # Logical coherence evaluation table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS logical_coherence_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    coherence_score REAL,
                    flow_score REAL,
                    has_argument_structure BOOLEAN,
                    has_discourse_markers BOOLEAN,
                    paragraph_score REAL,
                    rough_transitions TEXT,
                    total_sentences INTEGER,
                    total_paragraphs INTEGER
                )
            """)

            # Answer relevance evaluation table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS answer_relevance_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    relevance_score REAL,
                    semantic_similarity REAL,
                    entity_coverage REAL,
                    keyword_coverage REAL,
                    topic_focus REAL,
                    off_topic_sentences TEXT,
                    total_sentences INTEGER
                )
            """)
            
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
        If ID already exists, generates a new unique ID.

        Args:
            doc (ContentItem): A ContentItem instance containing the document data.

        Returns:
            bool: True if the document is new, False if updated
        """
        with self.lock:
            cursor = self.conn.cursor()
            try:
                # Check if document with this ID already exists
                cursor.execute("SELECT 1 FROM content WHERE id = ?", (doc.id,))
                id_exists = cursor.fetchone() is not None
                
                # If ID exists, generate a new unique ID
                if id_exists:
                    base_id = doc.id
                    counter = 1
                    while id_exists:
                        new_id = f"{base_id}_{counter}"
                        cursor.execute("SELECT 1 FROM content WHERE id = ?", (new_id,))
                        id_exists = cursor.fetchone() is not None
                        counter += 1
                    doc.id = new_id
                    logging.info(f"Generated new ID for document: {doc.id}")

                # Check if URL exists (for determining if this is an insert or update)
                cursor.execute("SELECT 1 FROM content WHERE url = ?", (doc.url,))
                is_new = cursor.fetchone() is None

                # Insert or update the document
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
                self.conn.rollback()
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

    def store_source_coverage(self, data: Dict[str, Any]) -> int:
        """
        Store source coverage evaluation results
        
        Args:
            data (Dict): Source coverage evaluation data
        
        Returns:
            int: ID of the inserted record
        """
        with self.lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO source_coverage_evaluations 
                    (query, coverage_score, coverage_ratio, diversity_score, missed_sources, total_sources)
                    VALUES (:query, :coverage_score, :coverage_ratio, :diversity_score, :missed_sources, :total_sources)
                    """,
                    {
                        'query': data.get('query', 'Unknown'),
                        'coverage_score': data.get('coverage_score', 0.0),
                        'coverage_ratio': data.get('coverage_ratio', 0.0),
                        'diversity_score': data.get('diversity_score', 0.0),
                        'missed_sources': json.dumps(data.get('missed_sources', [])),
                        'total_sources': data.get('total_sources', 0)
                    }
                )
                self.conn.commit()
                return cursor.lastrowid
            
            except sqlite3.Error as e:
                logging.error(f"Error storing source coverage: {e}")
                self.conn.rollback()
                return -1

    def store_logical_coherence(self, data: Dict[str, Any]) -> int:
        """
        Store logical coherence evaluation results
        
        Args:
            data (Dict): Logical coherence evaluation data
        
        Returns:
            int: ID of the inserted record
        """
        with self.lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO logical_coherence_evaluations 
                    (query, coherence_score, flow_score, has_argument_structure, 
                    has_discourse_markers, paragraph_score, rough_transitions, 
                    total_sentences, total_paragraphs)
                    VALUES (:query, :coherence_score, :flow_score, :has_argument_structure, 
                    :has_discourse_markers, :paragraph_score, :rough_transitions, 
                    :total_sentences, :total_paragraphs)
                    """,
                    {
                        'query': data.get('query', 'Unknown'),
                        'coherence_score': data.get('coherence_score', 0.0),
                        'flow_score': data.get('flow_score', 0.0),
                        'has_argument_structure': data.get('has_argument_structure', False),
                        'has_discourse_markers': data.get('has_discourse_markers', False),
                        'paragraph_score': data.get('paragraph_score', 0.0),
                        'rough_transitions': json.dumps(data.get('rough_transitions', [])),
                        'total_sentences': data.get('total_sentences', 0),
                        'total_paragraphs': data.get('total_paragraphs', 0)
                    }
                )
                self.conn.commit()
                return cursor.lastrowid
            
            except sqlite3.Error as e:
                logging.error(f"Error storing logical coherence: {e}")
                self.conn.rollback()
                return -1

    def store_answer_relevance(self, data: Dict[str, Any]) -> int:
        """
        Store answer relevance evaluation results
        
        Args:
            data (Dict): Answer relevance evaluation data
        
        Returns:
            int: ID of the inserted record
        """
        with self.lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO answer_relevance_evaluations 
                    (query, relevance_score, semantic_similarity, entity_coverage, 
                    keyword_coverage, topic_focus, off_topic_sentences, total_sentences)
                    VALUES (:query, :relevance_score, :semantic_similarity, :entity_coverage, 
                    :keyword_coverage, :topic_focus, :off_topic_sentences, :total_sentences)
                    """,
                    {
                        'query': data.get('query', 'Unknown'),
                        'relevance_score': data.get('relevance_score', 0.0),
                        'semantic_similarity': data.get('semantic_similarity', 0.0),
                        'entity_coverage': data.get('entity_coverage', 0.0),
                        'keyword_coverage': data.get('keyword_coverage', 0.0),
                        'topic_focus': data.get('topic_focus', 0.0),
                        'off_topic_sentences': json.dumps(data.get('off_topic_sentences', [])),
                        'total_sentences': data.get('total_sentences', 0)
                    }
                )
                self.conn.commit()
                return cursor.lastrowid
            
            except sqlite3.Error as e:
                logging.error(f"Error storing answer relevance: {e}")
                self.conn.rollback()
                return -1

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

    def get_source_coverage_evaluations(self, query: Optional[str] = None, limit: int = 10):
        """
        Retrieve source coverage evaluations
        
        Args:
            query (Optional[str]): Optional query to filter results
            limit (int): Maximum number of results to return
        
        Returns:
            List of source coverage evaluation dictionaries
        """
        with self.lock:
            cursor = self.conn.cursor()
            try:
                if query:
                    cursor.execute(
                        """
                        SELECT id, query, timestamp, coverage_score, coverage_ratio, 
                            diversity_score, missed_sources, total_sources
                        FROM source_coverage_evaluations
                        WHERE query LIKE ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (f'%{query}%', limit)
                    )
                else:
                    cursor.execute(
                        """
                        SELECT id, query, timestamp, coverage_score, coverage_ratio, 
                            diversity_score, missed_sources, total_sources
                        FROM source_coverage_evaluations
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (limit,)
                    )
                
                # Fetch and process results
                columns = [
                    'id', 'query', 'timestamp', 'coverage_score', 'coverage_ratio', 
                    'diversity_score', 'missed_sources', 'total_sources'
                ]
                results = []
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    # Parse JSON missed sources
                    result['missed_sources'] = json.loads(result['missed_sources']) if result['missed_sources'] else []
                    results.append(result)
                
                return results
            except Exception as e:
                logging.error(f"Error retrieving source coverage evaluations: {str(e)}")
                return []

def get_logical_coherence_evaluations(self, query: Optional[str] = None, limit: int = 10):
    """
    Retrieve logical coherence evaluations
    
    Args:
        query (Optional[str]): Optional query to filter results
        limit (int): Maximum number of results to return
    
    Returns:
        List of logical coherence evaluation dictionaries
    """
    with self.lock:
        cursor = self.conn.cursor()
        try:
            if query:
                cursor.execute(
                    """
                    SELECT id, query, timestamp, coherence_score, flow_score, 
                           has_argument_structure, has_discourse_markers, 
                           paragraph_score, rough_transitions, 
                           total_sentences, total_paragraphs
                    FROM logical_coherence_evaluations
                    WHERE query LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (f'%{query}%', limit)
                )
            else:
                cursor.execute(
                    """
                    SELECT id, query, timestamp, coherence_score, flow_score, 
                           has_argument_structure, has_discourse_markers, 
                           paragraph_score, rough_transitions, 
                           total_sentences, total_paragraphs
                    FROM logical_coherence_evaluations
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,)
                )
            
            # Fetch and process results
            columns = [
                'id', 'query', 'timestamp', 'coherence_score', 'flow_score', 
                'has_argument_structure', 'has_discourse_markers', 
                'paragraph_score', 'rough_transitions', 
                'total_sentences', 'total_paragraphs'
            ]
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                # Parse JSON rough transitions
                result['rough_transitions'] = json.loads(result['rough_transitions']) if result['rough_transitions'] else []
                results.append(result)
            
            return results
        except Exception as e:
            logging.error(f"Error retrieving logical coherence evaluations: {str(e)}")
            return []

def get_answer_relevance_evaluations(self, query: Optional[str] = None, limit: int = 10):
    """
    Retrieve answer relevance evaluations
    
    Args:
        query (Optional[str]): Optional query to filter results
        limit (int): Maximum number of results to return
    
    Returns:
        List of answer relevance evaluation dictionaries
    """
    with self.lock:
        cursor = self.conn.cursor()
        try:
            if query:
                cursor.execute(
                    """
                    SELECT id, query, timestamp, relevance_score, 
                           semantic_similarity, entity_coverage, 
                           keyword_coverage, topic_focus, 
                           off_topic_sentences, total_sentences
                    FROM answer_relevance_evaluations
                    WHERE query LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (f'%{query}%', limit)
                )
            else:
                cursor.execute(
                    """
                    SELECT id, query, timestamp, relevance_score, 
                           semantic_similarity, entity_coverage, 
                           keyword_coverage, topic_focus, 
                           off_topic_sentences, total_sentences
                    FROM answer_relevance_evaluations
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,)
                )
            
            # Fetch and process results
            columns = [
                'id', 'query', 'timestamp', 'relevance_score', 
                'semantic_similarity', 'entity_coverage', 
                'keyword_coverage', 'topic_focus', 
                'off_topic_sentences', 'total_sentences'
            ]
            results = []
            for row in cursor.fetchall():
                result = dict(zip(columns, row))
                # Parse JSON off-topic sentences
                result['off_topic_sentences'] = json.loads(result['off_topic_sentences']) if result['off_topic_sentences'] else []
                results.append(result)
            
            return results
        except Exception as e:
            logging.error(f"Error retrieving answer relevance evaluations: {str(e)}")
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