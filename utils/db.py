from tools.research.common.model_schemas import ContentItem
from typing import Optional, Dict, Any
import threading
import logging
import sqlite3
import os
import json
from datetime import datetime

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContentDB:
    def __init__(self, db_path: str):
          # Initial breakpoint
        self.lock = threading.Lock()
        
        db_dir = os.path.dirname(db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)

        with self.lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS content (
                    id TEXT PRIMARY KEY,
                    url TEXT UNIQUE,
                    title TEXT,
                    snippet TEXT,
                    content TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS automated_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    timestamp DATETIME,
                    overall_score REAL,
                    rouge1_score REAL,
                    rouge2_score REAL,
                    rougeL_score REAL,
                    semantic_similarity REAL,
                    hallucination_score REAL,
                    suspicious_segments TEXT
                );
                
                CREATE TABLE IF NOT EXISTS factual_accuracy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    timestamp TEXT,
                    factual_score REAL,
                    total_sources INTEGER,
                    citation_accuracy REAL,
                    claim_details TEXT,
                    contradicting_claims INTEGER,
                    verified_claims INTEGER,
                    unverified_claims INTEGER,
                    source_credibility_score REAL,
                    fact_check_coverage REAL
                );
                
                CREATE TABLE IF NOT EXISTS source_coverage_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    timestamp DATETIME,
                    coverage_score REAL,
                    coverage_ratio REAL,
                    diversity_score REAL,
                    missed_sources TEXT,
                    total_sources INTEGER,
                    unique_domains INTEGER,
                    source_depth REAL,
                    cross_referencing_score REAL,
                    domain_variety_score REAL
                );
                
                CREATE TABLE IF NOT EXISTS logical_coherence_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    timestamp DATETIME, 
                    coherence_score REAL,
                    flow_score REAL,
                    has_argument_structure BOOLEAN,
                    has_discourse_markers BOOLEAN,
                    paragraph_score REAL,
                    rough_transitions TEXT,
                    total_sentences INTEGER,
                    total_paragraphs INTEGER,
                    semantic_connection_score REAL,
                    idea_progression_score REAL,
                    logical_fallacies_count INTEGER,
                    topic_coherence REAL
                );
                
                CREATE TABLE IF NOT EXISTS answer_relevance_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    timestamp DATETIME,
                    relevance_score REAL,
                    semantic_similarity REAL,
                    entity_coverage REAL,
                    keyword_coverage REAL,
                    topic_focus REAL,
                    off_topic_sentences TEXT,
                    total_sentences INTEGER,
                    query_match_percentage REAL,
                    information_density REAL,
                    context_alignment_score REAL
                );
            """)
            self.conn.commit()
                
    def get_doc_by_id(self, id: str) -> Optional[ContentItem]:
          # Breakpoint before ID retrieval
        logger.info(f"Retrieving document with ID: {id}")
        
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
          # Breakpoint before URL retrieval
        logger.info(f"Retrieving document with URL: {url}")
        
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
          # Breakpoint before upsert
        logger.info(f"Upserting document: {doc.id}")
        
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
                    ON CONFLICT(id) DO UPDATE SET  
                        title=excluded.title,
                        snippet=excluded.snippet,
                        content=excluded.content,
                        source=excluded.source
                    """,
                    doc.to_dict(),
                )
                self.conn.commit()
                logging.info(f"Document {'inserted' if is_new else 'updated'} successfully: {doc.id}")
                  # Breakpoint after upsert
                return is_new
                
            except sqlite3.IntegrityError as e:
                logging.error(f"Error inserting/updating document: {e}")
                self.conn.rollback()
                raise

    def store_source_coverage(self, data: Dict[str, Any]) -> int:
          # Breakpoint before storing source coverage
        logger.info("Storing source coverage evaluation")
        
        with self.lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO source_coverage_evaluations 
                    (query, coverage_score, coverage_ratio, diversity_score, 
                        missed_sources, total_sources, unique_domains, source_depth, 
                        cross_referencing_score, domain_variety_score)
                    VALUES (:query, :coverage_score, :coverage_ratio, :diversity_score, 
                            :missed_sources, :total_sources, :unique_domains, :source_depth, 
                            :cross_referencing_score, :domain_variety_score)
                    """,
                    {
                        'query': data.get('query', 'Unknown'),
                        'coverage_score': data.get('coverage_score', 0.0),
                        'coverage_ratio': data.get('coverage_ratio', 0.0),
                        'diversity_score': data.get('diversity_score', 0.0),
                        'missed_sources': json.dumps(data.get('missed_sources', [])),
                        'total_sources': data.get('total_sources', 0),
                        'unique_domains': data.get('unique_domains', 0),
                        'source_depth': data.get('source_depth', 0.0),
                        'cross_referencing_score': data.get('cross_referencing_score', 0.0),
                        'domain_variety_score': data.get('domain_variety_score', 0.0)
                    }
                )
                self.conn.commit()
                  # Breakpoint after storing source coverage
                return cursor.lastrowid
            except sqlite3.Error as e:
                logging.error(f"Error storing source coverage: {e}")
                self.conn.rollback()
                return -1
            
    def store_test_results(self, query: str, overall_score: float, details: Dict[str, Any]) -> int:
        with self.lock:
            try:
                cursor = self.conn.execute("""
                    INSERT INTO automated_tests (
                        query, timestamp, overall_score, rouge1_score, rouge2_score, 
                        rougeL_score, semantic_similarity, hallucination_score, 
                        suspicious_segments
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    query,
                    details['timestamp'],
                    overall_score,
                    details['rouge_scores']['rouge1'],
                    details['rouge_scores']['rouge2'],
                    details['rouge_scores']['rougeL'],
                    details['semantic_similarity'],
                    details['hallucination_score'],
                    json.dumps(details['suspicious_segments'])
                ))
                self.conn.commit()
                return cursor.lastrowid
            except Exception as e:
                logging.error(f"Error storing test results: {e}")
                self.conn.rollback()
                return -1
                
    def store_logical_coherence(self, data: Dict[str, Any]) -> int:
          # Breakpoint before storing logical coherence
        logger.info("Storing logical coherence evaluation")
        
        with self.lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO logical_coherence_evaluations 
                    (query, coherence_score, flow_score, has_argument_structure, 
                     has_discourse_markers, paragraph_score, rough_transitions, 
                     total_sentences, total_paragraphs, semantic_connection_score, 
                     idea_progression_score, logical_fallacies_count)
                    VALUES (:query, :coherence_score, :flow_score, :has_argument_structure, 
                            :has_discourse_markers, :paragraph_score, :rough_transitions, 
                            :total_sentences, :total_paragraphs, :semantic_connection_score, 
                            :idea_progression_score, :logical_fallacies_count)
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
                        'total_paragraphs': data.get('total_paragraphs', 0),
                        'semantic_connection_score': data.get('semantic_connection_score', 0.0),
                        'idea_progression_score': data.get('idea_progression_score', 0.0),
                        'logical_fallacies_count': data.get('logical_fallacies_count', 0)
                    }
                )
                self.conn.commit()
                  # Breakpoint after storing logical coherence
                return cursor.lastrowid
            except sqlite3.Error as e:
                logging.error(f"Error storing logical coherence: {e}")
                self.conn.rollback()
                return -1

    def store_answer_relevance(self, data: Dict[str, Any]) -> int:
          # Breakpoint before storing answer relevance
        logger.info("Storing answer relevance evaluation")
        
        with self.lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO answer_relevance_evaluations 
                    (query, relevance_score, semantic_similarity, entity_coverage, 
                     keyword_coverage, topic_focus, off_topic_sentences, total_sentences,
                     query_match_percentage, information_density, context_alignment_score)
                    VALUES (:query, :relevance_score, :semantic_similarity, :entity_coverage, 
                            :keyword_coverage, :topic_focus, :off_topic_sentences, :total_sentences,
                            :query_match_percentage, :information_density, :context_alignment_score)
                    """,
                    {
                        'query': data.get('query', 'Unknown'),
                        'relevance_score': data.get('relevance_score', 0.0),
                        'semantic_similarity': data.get('semantic_similarity', 0.0),
                        'entity_coverage': data.get('entity_coverage', 0.0),
                        'keyword_coverage': data.get('keyword_coverage', 0.0),
                        'topic_focus': data.get('topic_focus', 0.0),
                        'off_topic_sentences': json.dumps(data.get('off_topic_sentences', [])),
                        'total_sentences': data.get('total_sentences', 0),
                        'query_match_percentage': data.get('query_match_percentage', 0.0),
                        'information_density': data.get('information_density', 0.0),
                        'context_alignment_score': data.get('context_alignment_score', 0.0)
                    }
                )
                self.conn.commit()
                  # Breakpoint after storing answer relevance
                return cursor.lastrowid
            except sqlite3.Error as e:
                logging.error(f"Error storing answer relevance: {e}")
                self.conn.rollback()
                return -1
                
    def store_accuracy_evaluation(self, accuracy_data: Dict[str, Any]):
          # Breakpoint before storing accuracy evaluation
        logger.info("Storing accuracy evaluation")
        
        with self.lock:
            try:
                insert_data = {
                    'query': accuracy_data.get('query', 'Unknown'),
                    'timestamp': accuracy_data.get('timestamp', datetime.now().isoformat()),
                    'factual_score': accuracy_data.get('factual_score', 0.0),
                    'total_sources': accuracy_data.get('total_sources', 0),
                    'citation_accuracy': accuracy_data.get('citation_accuracy', 0.0),
                    'claim_details': json.dumps(accuracy_data.get('claim_details', [])),
                    'contradicting_claims': accuracy_data.get('contradicting_claims', 0),
                    'verified_claims': accuracy_data.get('verified_claims', 0),
                    'unverified_claims': accuracy_data.get('unverified_claims', 0),
                    'source_credibility_score': accuracy_data.get('source_credibility_score', 0.0),
                    'fact_check_coverage': accuracy_data.get('fact_check_coverage', 0.0)
                }

                cursor = self.conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO factual_accuracy 
                    (query, timestamp, factual_score, total_sources, 
                    citation_accuracy, claim_details, contradicting_claims,
                    verified_claims, unverified_claims, source_credibility_score,
                    fact_check_coverage)
                    VALUES (:query, :timestamp, :factual_score, :total_sources, 
                            :citation_accuracy, :claim_details, :contradicting_claims,
                            :verified_claims, :unverified_claims, :source_credibility_score,
                            :fact_check_coverage)
                    """,
                    insert_data
                )
                
                self.conn.commit()
                  # Breakpoint after storing accuracy evaluation
                return cursor.lastrowid
            except Exception as e:
                logger.error(f"Error storing factual accuracy: {str(e)}")
                self.conn.rollback()
                return None
    def get_test_results(self, query: str = None, limit: int = 50):
        with self.lock:
            try:
                if query:
                    cursor = self.conn.execute(
                        """
                        SELECT * FROM automated_tests 
                        WHERE query LIKE ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                        """,
                        (f'%{query}%', limit)
                    )
                else:
                    cursor = self.conn.execute(
                        """
                        SELECT * FROM automated_tests 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                        """,
                        (limit,)
                    )
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    result['suspicious_segments'] = json.loads(result['suspicious_segments'])
                    results.append(result)
                return results
            except Exception as e:
                logging.error(f"Error retrieving test results: {e}")
                return []
    def get_accuracy_evaluations(self, query: Optional[str] = None, limit: int = 10):
        # Breakpoint before retrieving accuracy evaluations
        logger.info(f"Retrieving accuracy evaluations for query: {query}")
        
        with self.lock:
            cursor = self.conn.cursor()
            try:
                if query:
                    cursor.execute(
                        """
                        SELECT id, query, timestamp, factual_score, total_sources, 
                        citation_accuracy, claim_details, contradicting_claims,
                        verified_claims, unverified_claims, 
                        source_credibility_score, fact_check_coverage
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
                        citation_accuracy, claim_details, contradicting_claims,
                        verified_claims, unverified_claims, 
                        source_credibility_score, fact_check_coverage
                        FROM factual_accuracy 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                        """, 
                        (limit,)
                    )
                
                columns = [
                    'id', 'query', 'timestamp', 'factual_score', 'total_sources', 
                    'citation_accuracy', 'claim_details', 'contradicting_claims',
                    'verified_claims', 'unverified_claims', 
                    'source_credibility_score', 'fact_check_coverage'
                ]
                results = []
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    result['claim_details'] = json.loads(result['claim_details']) if result['claim_details'] else []
                    results.append(result)
                
                # Breakpoint after retrieving accuracy evaluations
                return results
            except Exception as e:
                logger.error(f"Error retrieving accuracy evaluations: {str(e)}")
                return []

    def get_source_coverage_evaluations(self, query: Optional[str] = None, limit: int = 10):
        # Breakpoint before retrieving source coverage evaluations
        logger.info(f"Retrieving source coverage evaluations for query: {query}")
        
        with self.lock:
            cursor = self.conn.cursor()
            try:
                if query:
                    cursor.execute(
                        """
                        SELECT id, query, timestamp, coverage_score, coverage_ratio, 
                            diversity_score, missed_sources, total_sources,
                            unique_domains, source_depth, 
                            cross_referencing_score, domain_variety_score
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
                            diversity_score, missed_sources, total_sources,
                            unique_domains, source_depth, 
                            cross_referencing_score, domain_variety_score
                        FROM source_coverage_evaluations
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (limit,)
                    )
                
                columns = [
                    'id', 'query', 'timestamp', 'coverage_score', 'coverage_ratio', 
                    'diversity_score', 'missed_sources', 'total_sources',
                    'unique_domains', 'source_depth', 
                    'cross_referencing_score', 'domain_variety_score'
                ]
                results = []
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    result['missed_sources'] = json.loads(result['missed_sources']) if result['missed_sources'] else []
                    results.append(result)
                
                # Breakpoint after retrieving source coverage evaluations
                return results
            except Exception as e:
                logger.error(f"Error retrieving source coverage evaluations: {str(e)}")
                return []

    def get_logical_coherence_evaluations(self, query: Optional[str] = None, limit: int = 10):
    # Breakpoint before retrieving logical coherence evaluations
        logger.info(f"Retrieving logical coherence evaluations for query: {query}")
        
        with self.lock:
            cursor = self.conn.cursor()
            try:
                if query:
                    cursor.execute(
                        """
                        SELECT id, query, timestamp, coherence_score, flow_score, 
                        has_argument_structure, has_discourse_markers, 
                        paragraph_score, rough_transitions, 
                        total_sentences, total_paragraphs,
                        semantic_connection_score, idea_progression_score,
                        logical_fallacies_count
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
                        total_sentences, total_paragraphs,
                        semantic_connection_score, idea_progression_score,
                        logical_fallacies_count
                        FROM logical_coherence_evaluations
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (limit,)
                    )
                
                columns = [
                    'id', 'query', 'timestamp', 'coherence_score', 'flow_score', 
                    'has_argument_structure', 'has_discourse_markers', 
                    'paragraph_score', 'rough_transitions', 
                    'total_sentences', 'total_paragraphs',
                    'semantic_connection_score', 'idea_progression_score',
                    'logical_fallacies_count'
                ]
                results = []
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    result['rough_transitions'] = json.loads(result['rough_transitions']) if result['rough_transitions'] else []
                    results.append(result)
                
                # Breakpoint after retrieving logical coherence evaluations
                return results
            except Exception as e:
                logger.error(f"Error retrieving logical coherence evaluations: {str(e)}")
                return []

    def get_answer_relevance_evaluations(self, query: Optional[str] = None, limit: int = 10):
        # Breakpoint before retrieving answer relevance evaluations
        logger.info(f"Retrieving answer relevance evaluations for query: {query}")
        
        with self.lock:
            cursor = self.conn.cursor()
            try:
                if query:
                    cursor.execute(
                        """
                        SELECT id, query, timestamp, relevance_score, semantic_similarity, 
                            entity_coverage, keyword_coverage, topic_focus, 
                            off_topic_sentences, total_sentences, query_match_percentage, 
                            information_density, context_alignment_score
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
                        SELECT id, query, timestamp, relevance_score, semantic_similarity, 
                            entity_coverage, keyword_coverage, topic_focus, 
                            off_topic_sentences, total_sentences, query_match_percentage, 
                            information_density, context_alignment_score
                        FROM answer_relevance_evaluations
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """, 
                        (limit,)
                    )
                
                columns = [
                    'id', 'query', 'timestamp', 'relevance_score', 'semantic_similarity', 
                    'entity_coverage', 'keyword_coverage', 'topic_focus', 
                    'off_topic_sentences', 'total_sentences', 'query_match_percentage', 
                    'information_density', 'context_alignment_score'
                ]
                results = []
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    result['off_topic_sentences'] = json.loads(result['off_topic_sentences']) if result['off_topic_sentences'] else []
                    results.append(result)
                
                # Breakpoint after retrieving answer relevance evaluations
                return results
            except Exception as e:
                logger.error(f"Error retrieving answer relevance evaluations: {str(e)}")
                return []
                
    def delete_doc(self, id: str):
    # Breakpoint before document deletion
        logger.info(f"Deleting document with ID: {id}")
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM content WHERE id = ?", (id,))
            self.conn.commit()

    def generate_snippet(self, text: str) -> str:
    # Breakpoint before snippet generation
        logger.info(f"Generating snippet for text of length {len(text)}")

        return f"{text[:150]}..." 

    def close(self):
    # Breakpoint before closing connection
        logger.info("Closing database connection")
        
        with self.lock:
            self.conn.close()