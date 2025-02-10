import logging
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.research.common.model_schemas import ResearchToolOutput, ContentItem
from typing import List, Dict, Tuple, Union, Any, Optional
import json
import os
from typing import Optional
import sys
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime
from collections import Counter
from langfuse import Langfuse
from concurrent.futures import ThreadPoolExecutor
import asyncio

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d')
    log_file = os.path.join(log_dir, f'relevance_evaluator_{timestamp}.log')
    
    logger = logging.getLogger('relevance_evaluator')
    logger.setLevel(logging.DEBUG)
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

try:
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    logger.info("Successfully initialized spaCy model")
except Exception as e:
    logger.critical(f"Failed to initialize spaCy model: {str(e)}", exc_info=True)
    raise

class LangfuseTracker:
    def __init__(self, 
                public_key: Optional[str] = None, 
                secret_key: Optional[str] = None, 
                host: Optional[str] = None):
       # Use environment variables if not explicitly provided
       public_key = public_key or os.getenv('LANGFUSE_PUBLIC_KEY')
       secret_key = secret_key or os.getenv('LANGFUSE_SECRET_KEY')
       host = host or os.getenv('LANGFUSE_HOST')

       if not public_key or not secret_key:
           raise ValueError("Langfuse public and secret keys must be provided")

       self.langfuse = Langfuse(
           public_key=public_key, 
           secret_key=secret_key, 
           host=host
       )
       self._executor = ThreadPoolExecutor(max_workers=4)

    async def track_relevance_metric(self, trace_id: str, 
                                   score: float, details: Dict[str, Any]) -> None:
        """Track relevance evaluation metrics in Langfuse."""
        try:
            generation = self.langfuse.get_generation(trace_id)
            generation.score(
                name="answer-relevance",
                value=score,
                metadata={
                    'semantic_similarity': details.get('semantic_similarity', 0.0),
                    'keyword_coverage': details.get('keyword_coverage', 0.0),
                    'entity_coverage': details.get('entity_coverage', 0.0),
                    'information_density': details.get('information_density', 0.0),
                    'context_alignment': details.get('context_alignment_score', 0.0),
                    'total_sentences': details.get('total_sentences', 0),
                    'off_topic_sentences': len(details.get('off_topic_sentences', [])),
                    'timestamp': datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error tracking relevance metrics: {str(e)}", exc_info=True)

    def track_error(self, trace_id: str, error: str) -> None:
        """Track evaluation errors in Langfuse."""
        try:
            trace = self.langfuse.get_trace(trace_id)
            trace.error(
                error=error,
                metadata={
                    "component": "answer_relevance",
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error tracking error event: {str(e)}", exc_info=True)

class AnswerRelevanceEvaluator:
    def __init__(self, langfuse_tracker: Optional[LangfuseTracker] = None):
        self.logger = logging.getLogger('relevance_evaluator.evaluator')
        self.langfuse_tracker = langfuse_tracker
        
        try:
            self.nlp = nlp
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                strip_accents='unicode'
            )
        except Exception as e:
            self.logger.error("Failed to initialize components", exc_info=True)
            raise

    def calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            vectors = self.vectorizer.fit_transform([text1, text2])
            return float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}", exc_info=True)
            return 0.0

    def _extract_text_from_output(self, research_output: Union[ResearchToolOutput, str, List]) -> str:
        try:
            if isinstance(research_output, str):
                return research_output
            if isinstance(research_output, ResearchToolOutput):
                return research_output.get_full_text()
            if isinstance(research_output, list):
                texts = []
                for item in research_output:
                    if hasattr(item, 'text'):
                        texts.append(item.text)
                    elif isinstance(item, str):
                        texts.append(item)
                    else:
                        texts.append(str(item))
                return " ".join(texts)
            if hasattr(research_output, 'text'):
                return research_output.text
            return str(research_output)
        except Exception as e:
            self.logger.error(f"Error extracting text: {str(e)}", exc_info=True)
            return ""

    async def evaluate_answer_relevance(
        self,
        research_output: Union[ResearchToolOutput, str],
        query: str,
        trace_id: Optional[str] = None
    ) -> Tuple[float, Dict]:
        try:
            full_text = self._extract_text_from_output(research_output)
            if not full_text.strip():
                return 0.0, self._create_empty_evaluation()

            query_doc = self.nlp(query)
            answer_doc = self.nlp(full_text)

            similarity = self.calculate_similarity(query, full_text)
            
            query_keywords = {token.lemma_.lower() for token in query_doc 
                            if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop}
            answer_keywords = {token.lemma_.lower() for token in answer_doc 
                             if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop}
            
            keyword_coverage = (len(query_keywords.intersection(answer_keywords)) / len(query_keywords) 
                              if query_keywords else 0)
            
            query_entities = set(ent.text.lower() for ent in query_doc.ents)
            answer_entities = set(ent.text.lower() for ent in answer_doc.ents)
            entity_coverage = len(query_entities.intersection(answer_entities))
            
            sentences = list(answer_doc.sents)
            sentence_similarities = [
                (sent.text, self.calculate_similarity(query, sent.text))
                for sent in sentences
            ]
            
            off_topic_sentences = [
                sent_text for sent_text, sent_sim in sentence_similarities
                if sent_sim < 0.2
            ]
            
            words = full_text.split()
            information_density = len(answer_keywords) / len(words) if words else 0
            
            off_topic_ratio = len(off_topic_sentences) / len(sentences) if sentences else 0
            context_alignment_score = similarity * (1 - off_topic_ratio)
            
            relevance_score = (
                0.4 * similarity +
                0.3 * keyword_coverage +
                0.3 * context_alignment_score
            )
            
            evaluation_result = {
                'relevance_score': relevance_score,
                'semantic_similarity': similarity,
                'entity_coverage': entity_coverage,
                'keyword_coverage': keyword_coverage,
                'topic_focus': similarity,
                'off_topic_sentences': off_topic_sentences,
                'total_sentences': len(sentences),
                'query_match_percentage': relevance_score * 100,
                'information_density': information_density,
                'context_alignment_score': context_alignment_score
            }

            if trace_id and self.langfuse_tracker:
                await self.langfuse_tracker.track_relevance_metric(
                    trace_id, relevance_score, evaluation_result
                )

            return relevance_score, evaluation_result

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error in evaluation: {error_msg}", exc_info=True)
            if trace_id and self.langfuse_tracker:
                self.langfuse_tracker.track_error(trace_id, error_msg)
            return 0.0, self._create_empty_evaluation()

    def _create_empty_evaluation(self) -> Dict:
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

def create_answer_relevance_evaluator(
   langfuse_public_key: Optional[str] = None,
   langfuse_secret_key: Optional[str] = None,
   langfuse_host: Optional[str] = None
) -> AnswerRelevanceEvaluator:
   # Use environment variables if not explicitly provided
   langfuse_public_key = langfuse_public_key or os.getenv('LANGFUSE_PUBLIC_KEY')
   langfuse_secret_key = langfuse_secret_key or os.getenv('LANGFUSE_SECRET_KEY')
   langfuse_host = langfuse_host or os.getenv('LANGFUSE_HOST')

   langfuse_tracker = None
   if langfuse_public_key and langfuse_secret_key:
       try:
           # Create configuration dictionary
           langfuse_config = {
               'public_key': langfuse_public_key,
               'secret_key': langfuse_secret_key
           }
           if langfuse_host:
               langfuse_config['host'] = langfuse_host

           # Pass configuration to LangfuseTracker
           langfuse_tracker = LangfuseTracker(
               public_key=langfuse_public_key,
               secret_key=langfuse_secret_key,
               host=langfuse_host
           )
           logger.info("Langfuse integration enabled")
       except Exception as e:
           logger.error(f"Failed to initialize Langfuse: {str(e)}")
   
   return AnswerRelevanceEvaluator(langfuse_tracker=langfuse_tracker)