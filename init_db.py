import os
import logging
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)
import sqlite3
import json
from datetime import datetime
from dotenv import load_dotenv
from research_components.db import ContentDB
from tools.research.common.model_schemas import ContentItem

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_database(db: ContentDB):
    try:
        with db.conn:
            # Database schema
            db.conn.executescript("""
                -- Core tables
                CREATE TABLE IF NOT EXISTS content (
                    id TEXT PRIMARY KEY,
                    url TEXT UNIQUE,
                    title TEXT,
                    snippet TEXT,
                    content TEXT,
                    source TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    duration REAL,
                    success BOOLEAN,
                    error TEXT,
                    data TEXT,
                    analysis_metrics TEXT
                );

                -- Analysis and Evaluation tables
                CREATE TABLE IF NOT EXISTS analysis_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    timestamp TEXT,
                    analysis TEXT,
                    numerical_accuracy REAL,
                    query_understanding REAL,
                    data_validation REAL,
                    reasoning_transparency REAL,
                    overall_score REAL,
                    metrics_details TEXT,
                    calculation_examples TEXT,
                    term_coverage REAL,
                    analytical_elements TEXT,
                    validation_checks TEXT,
                    explanation_patterns TEXT
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

                -- Coverage and Coherence tables
                CREATE TABLE IF NOT EXISTS source_coverage_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    timestamp DATETIME,
                    coverage_score REAL,
                    coverage_ratio REAL,
                    semantic_coverage REAL,         
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
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
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

                -- Metrics and Performance tables
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id INTEGER,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(trace_id) REFERENCES traces(id)
                );

                CREATE TABLE IF NOT EXISTS analysis_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    overall_score REAL,
                    processing_time REAL,
                    accuracy_score REAL,
                    completeness_score REAL,
                    relevance_score REAL,
                    details TEXT
                );

                CREATE TABLE IF NOT EXISTS evaluation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id INTEGER,
                    type TEXT,
                    score REAL,
                    details TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(trace_id) REFERENCES traces(id)
                );

                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    success_rate REAL,
                    avg_score REAL,
                    total_analyses INTEGER,
                    avg_duration REAL,
                    FOREIGN KEY(id) REFERENCES traces(id)
                );

                -- Query and Content Results tables
                CREATE TABLE IF NOT EXISTS query_traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT,
                    tool TEXT,
                    prompt_name TEXT,
                    tools_used TEXT,
                    processing_steps TEXT,
                    duration REAL,
                    success BOOLEAN,
                    content_new INTEGER,
                    content_reused INTEGER,
                    error_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS content_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id INTEGER,
                    trace_id INTEGER,
                    content TEXT,
                    content_type TEXT,
                    evaluation_metrics TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(trace_id) REFERENCES query_traces(id)
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

                CREATE TABLE IF NOT EXISTS result_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id INTEGER,
                    metric_type TEXT,
                    metric_value REAL,
                    metric_details TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(trace_id) REFERENCES traces(id)
                );

                -- Indexes for optimization
                CREATE INDEX IF NOT EXISTS idx_traces_query ON traces(query);
                CREATE INDEX IF NOT EXISTS idx_analysis_metrics_query ON analysis_metrics(query);
                CREATE INDEX IF NOT EXISTS idx_result_metrics_trace ON result_metrics(trace_id);
                CREATE INDEX IF NOT EXISTS idx_metrics_trace ON metrics(trace_id);
                CREATE INDEX IF NOT EXISTS idx_eval_metrics_trace ON evaluation_metrics(trace_id);
                CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_metrics(date);
                CREATE INDEX IF NOT EXISTS idx_content_results_trace ON content_results(trace_id);
                CREATE INDEX IF NOT EXISTS idx_query_traces_timestamp ON query_traces(timestamp);
            """)

        db_size = os.path.getsize(db_path)
        if db_size == 0:
            logger.info(f"Adding sample document to {db_path}")  
            sample_doc = ContentItem(...)
            db.upsert_doc(sample_doc)
            logger.info("Sample document added to database")
            
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise

if __name__ == "__main__":
    db_path = os.getenv('DB_PATH', './data/content.db') 
    init_db = os.getenv('INIT_DB', 'false').lower() == 'true'
    
    db = ContentDB(db_path)
    
    if init_db:
        logger.info("INIT_DB is true. Running database initialization.") 
        initialize_database(db)
    else:
        logger.info("INIT_DB is false. Skipping database initialization.")