import logging
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.research.common.model_schemas import ResearchToolOutput, ContentItem
from typing import List, Dict, Tuple, Union, Any
import json
import sys
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from collections import Counter

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Sets up logging configuration with both file and console handlers.
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f'relevance_evaluator_{timestamp}.log')
    
    # Create logger
    logger = logging.getLogger('relevance_evaluator')
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Create rotating file handler (10MB per file, max 5 backup files)
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Load spaCy model with error handling
try:
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    logger.info("Successfully initialized spaCy model")
except Exception as e:
    logger.critical(f"Failed to initialize spaCy model: {str(e)}", exc_info=True)
    raise

class AnswerRelevanceEvaluator:
    """
    Evaluates the relevance of research outputs in relation to input queries.
    Handles both ResearchToolOutput objects and raw strings.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('relevance_evaluator.evaluator')
        self.logger.info("Initializing AnswerRelevanceEvaluator")
        try:
            self.logger.debug("Setting up spaCy pipeline")
            self.nlp = nlp
            self.logger.debug(f"SpaCy pipeline components: {', '.join(self.nlp.pipe_names)}")
            
            self.logger.debug("Configuring TF-IDF vectorizer")
            self.vectorizer = TfidfVectorizer(
                max_features=5000, 
                stop_words='english',
                strip_accents='unicode'
            )
            self.logger.debug("Successfully initialized TfidfVectorizer")
            self.logger.debug(f"Vectorizer parameters: {self.vectorizer.get_params()}")
            
        except Exception as e:
            self.logger.error("Failed to initialize components", exc_info=True)
            raise
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts using TF-IDF vectors.
        """
        self.logger.debug("Calculating similarity between texts")
        self.logger.debug(f"Text1 length: {len(text1)}, Text2 length: {len(text2)}")
        self.logger.debug(f"Text1 sample: {text1[:50]}...")
        self.logger.debug(f"Text2 sample: {text2[:50]}...")
        self.logger.debug("Vectorizer config: max_features=5000, using english stop words")
        
        if not text1.strip() or not text2.strip():
            self.logger.warning("Empty text provided for similarity calculation")
            return 0.0
            
        try:
            vectors = self.vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            self.logger.debug(f"Similarity calculation successful: {similarity:.4f}")
            return similarity
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}", exc_info=True)
            return 0.0
    
    def _extract_text_from_output(self, research_output: Union[ResearchToolOutput, str, List]) -> str:
        """
        Extract text content from research output or return the string directly.
        Handles string, ResearchToolOutput, ContentItem, and list inputs.
        """
        self.logger.debug(f"Extracting text from output of type: {type(research_output)}")
        
        try:
            if isinstance(research_output, str):
                self.logger.debug("Processing string input")
                return research_output
                
            if isinstance(research_output, ResearchToolOutput):
                self.logger.debug("Processing ResearchToolOutput")
                return research_output.get_full_text()
                
            if isinstance(research_output, list):
                self.logger.debug(f"Processing list input with {len(research_output)} items")
                # Handle list of ContentItems or strings
                texts = []
                for item in research_output:
                    if hasattr(item, 'text'):
                        self.logger.debug(f"Extracting text from ContentItem: {type(item)}")
                        texts.append(item.text)
                    elif isinstance(item, str):
                        texts.append(item)
                    else:
                        self.logger.warning(f"Unexpected item type in list: {type(item)}")
                        texts.append(str(item))
                return " ".join(texts)
            
            # Handle single ContentItem
            if hasattr(research_output, 'text'):
                self.logger.debug("Processing ContentItem")
                return research_output.text
            
            self.logger.warning(f"Unexpected input type: {type(research_output)}")
            return str(research_output)
            
        except Exception as e:
            self.logger.error(
                f"Error extracting text from {type(research_output)}: {str(e)}", 
                exc_info=True,
                extra={'research_output_type': str(type(research_output))}
            )
            return ""

    def evaluate_answer_relevance(self, research_output: Union[ResearchToolOutput, str], query: str) -> Tuple[float, Dict]:
        """
        Evaluates the relevance of a research output to a query.
        """
        self.logger.info(f"Starting relevance evaluation for query: {query[:100]}...")
        self.logger.debug(f"Full query length: {len(query)}")
        
        try:
            full_text = self._extract_text_from_output(research_output)
            
            if not full_text.strip():
                self.logger.warning("Empty text extracted from research output")
                return 0.0, self._create_empty_evaluation()
                
            self.logger.debug(f"Extracted text length: {len(full_text)}")
            
            # Process documents with spaCy
            self.logger.debug("Processing documents with spaCy")
            self.logger.debug(f"Query processing started: {query[:100]}...")
            query_doc = self.nlp(query)
            self.logger.debug(f"Query tokens: {[token.text for token in query_doc][:10]}")
            
            self.logger.debug("Processing answer document")
            answer_doc = self.nlp(full_text)
            self.logger.debug(f"Answer document length: {len(answer_doc)}, sentences: {len(list(answer_doc.sents))}")
            self.logger.debug(f"First few tokens: {[token.text for token in answer_doc][:10]}")
            
            # Calculate semantic similarity
            similarity = self.calculate_similarity(query, full_text)
            self.logger.info(f"Semantic similarity score: {similarity:.4f}")
            
            # Calculate keyword coverage
            query_keywords = {token.lemma_.lower() for token in query_doc 
                            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop}
            answer_keywords = {token.lemma_.lower() for token in answer_doc 
                             if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop}
            
            self.logger.debug(f"Query keywords found: {len(query_keywords)}")
            self.logger.debug(f"Query keywords: {list(query_keywords)[:10]}")
            self.logger.debug(f"Answer keywords found: {len(answer_keywords)}")
            self.logger.debug(f"Common keywords: {list(query_keywords.intersection(answer_keywords))}")
            self.logger.debug(f"POS distribution in query: {dict(Counter(token.pos_ for token in query_doc))}")
            self.logger.debug(f"POS distribution in answer: {dict(Counter(token.pos_ for token in answer_doc))}")
            
            
            keyword_coverage = (len(query_keywords.intersection(answer_keywords)) / len(query_keywords) 
                              if query_keywords else 0)
            self.logger.info(f"Keyword coverage score: {keyword_coverage:.4f}")
            
            # Entity coverage calculation
            query_entities = set(ent.text.lower() for ent in query_doc.ents)
            answer_entities = set(ent.text.lower() for ent in answer_doc.ents)
            entity_coverage = len(query_entities.intersection(answer_entities))
            
            self.logger.debug(f"Query entities found: {list(query_entities)}")
            self.logger.debug(f"Answer entities found: {list(answer_entities)[:10]}")
            self.logger.debug(f"Matching entities: {list(query_entities.intersection(answer_entities))}")
            self.logger.debug(f"Entity types in query: {[ent.label_ for ent in query_doc.ents]}")
            self.logger.debug(f"Entity coverage count: {entity_coverage}")
            self.logger.debug(f"Entity coverage percentage: {(entity_coverage / len(query_entities) * 100) if query_entities else 0:.2f}%")
            
            # Process sentences
            sentences = list(answer_doc.sents)
            self.logger.debug(f"Processing {len(sentences)} sentences")
            self.logger.debug(f"Average sentence length: {sum(len(sent) for sent in sentences) / len(sentences) if sentences else 0:.1f} tokens")
            self.logger.debug(f"First 3 sentences: {[sent.text for sent in sentences[:3]]}")
            self.logger.debug(f"Sentence length distribution: {[(i, len([s for s in sentences if len(s) == i])) for i in range(1, 6)]}")
            
            sentence_similarities = [
                (sent.text, self.calculate_similarity(query, sent.text))
                for sent in sentences
            ]
            
            # Identify off-topic sentences
            off_topic_threshold = 0.2
            off_topic_sentences = [
                sent_text for sent_text, sent_sim in sentence_similarities
                if sent_sim < off_topic_threshold
            ]
            
            total_sentences = len(sentences)
            self.logger.info(
                f"Sentence analysis - Total: {total_sentences}, "
                f"Off-topic: {len(off_topic_sentences)}"
            )
            
            # Calculate information density
            words = full_text.split()
            information_density = len(answer_keywords) / len(words) if words else 0
            self.logger.debug(f"Information density: {information_density:.4f}")
            
            # Calculate context alignment score
            off_topic_ratio = len(off_topic_sentences) / total_sentences if total_sentences > 0 else 0
            context_alignment_score = similarity * (1 - off_topic_ratio)
            self.logger.debug(f"Context alignment score: {context_alignment_score:.4f}")
            
            # Calculate overall relevance score with weighting
            weights = {
                'similarity': 0.4,
                'keyword_coverage': 0.3,
                'context_alignment': 0.3
            }
            
            relevance_score = (
                weights['similarity'] * similarity +
                weights['keyword_coverage'] * keyword_coverage +
                weights['context_alignment'] * context_alignment_score
            )
            
            evaluation_result = {
                'relevance_score': relevance_score,
                'semantic_similarity': similarity,
                'entity_coverage': entity_coverage,
                'keyword_coverage': keyword_coverage,
                'topic_focus': similarity,
                'off_topic_sentences': off_topic_sentences,
                'total_sentences': total_sentences,
                'query_match_percentage': relevance_score * 100,
                'information_density': information_density,
                'context_alignment_score': context_alignment_score
            }
            
            self.logger.info(f"Completed relevance evaluation with score: {relevance_score:.4f}")
            return relevance_score, evaluation_result
            
        except Exception as e:
            self.logger.error(
                f"Error in evaluate_answer_relevance: {str(e)}", 
                exc_info=True,
                extra={'query': query[:100]}
            )
            return 0.0, self._create_empty_evaluation()
    
    def _create_empty_evaluation(self) -> Dict:
        """Create an empty evaluation result with zero values."""
        self.logger.warning("Creating empty evaluation result")
        return {
            'relevance_score': 0.0,
            'semantic_similarity': 0.0,
            'entity_coverage': 0.0,
            'keyword_coverage': 0.0,
            'topic_focus': 0.0,
            'off_topic_sentences': [],
            'total_sentences': 0,
            'query_match_percentage': 0.0,
            'information_density': 0.0,
            'context_alignment_score': 0.0
        }

def create_answer_relevance_evaluator() -> AnswerRelevanceEvaluator:
    """Factory function to create an AnswerRelevanceEvaluator instance."""
    logger.info("Creating new AnswerRelevanceEvaluator instance")
    return AnswerRelevanceEvaluator()