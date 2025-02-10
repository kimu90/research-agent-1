import spacy
import numpy as np
import urllib.parse
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.research.common.model_schemas import ResearchToolOutput
from itertools import combinations
from langfuse import Langfuse
from concurrent.futures import ThreadPoolExecutor
import asyncio
from datetime import datetime
import logging
import os
from typing import Optional
logger = logging.getLogger(__name__)

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

    async def track_coverage_metric(self, trace_id: str, score: float, details: Dict) -> None:
        try:
            generation = self.langfuse.get_generation(trace_id)
            generation.score(
                name="source-coverage",
                value=score,
                metadata={
                    'coverage_ratio': details.get('coverage_ratio', 0.0),
                    'semantic_coverage': details.get('semantic_coverage', 0.0),
                    'total_sources': details.get('total_sources', 0),
                    'unique_domains': details.get('unique_domains', 0),
                    'source_depth': details.get('source_depth', 0.0),
                    'cross_referencing': details.get('cross_referencing_score', 0.0),
                    'domain_variety': details.get('domain_variety_score', 0.0),
                    'timestamp': datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error tracking coverage metrics: {str(e)}", exc_info=True)

    def track_error(self, trace_id: str, error: str) -> None:
        try:
            trace = self.langfuse.get_trace(trace_id)
            trace.error(
                error=error,
                metadata={
                    "component": "source_coverage",
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error tracking error event: {str(e)}", exc_info=True)

class SourceCoverageEvaluator:
    def __init__(self, langfuse_tracker: Optional[LangfuseTracker] = None):
        self.nlp = spacy.blank('en')
        self.nlp.add_pipe('sentencizer')
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.langfuse_tracker = langfuse_tracker
        logger.info("Initialized SourceCoverageEvaluator")

    def calculate_source_depth(self, url: str) -> int:
        try:
            return len(urllib.parse.urlparse(url).path.strip('/').split('/'))
        except Exception:
            return 0

    async def evaluate_source_coverage(
        self, 
        research_output: ResearchToolOutput,
        trace_id: Optional[str] = None
    ) -> Tuple[float, Dict]:
        try:
            all_sources = [
                {'url': content.url, 'text': content.content or content.snippet} 
                for content in research_output.content
            ]
            
            unique_domains = set(
                urllib.parse.urlparse(source['url']).netloc 
                for source in all_sources if source['url']
            )
            
            source_depths = [self.calculate_source_depth(source['url']) for source in all_sources]
            source_depth = np.mean(source_depths) if source_depths else 0
            
            all_texts = [source['text'] for source in all_sources if source['text']]
            
            if all_texts:
                self.vectorizer.fit(all_texts)
                vectors = self.vectorizer.transform(all_texts)
                semantic_similarity_matrix = cosine_similarity(vectors)
                
                mask = np.triu(np.ones_like(semantic_similarity_matrix), k=1)
                semantic_coverage = np.mean(semantic_similarity_matrix[mask > 0]) if len(all_texts) > 1 else 0
                
                pairs = list(combinations(range(len(all_texts)), 2))
                cross_references = sum(semantic_similarity_matrix[i][j] > 0.3 for i, j in pairs)
                cross_referencing_score = cross_references / len(pairs) if pairs else 0
            else:
                semantic_coverage = cross_referencing_score = 0
            
            domain_variety_score = len(unique_domains) / len(all_sources) if all_sources else 0
            normalized_depth = min(source_depth / 5, 1.0)
            coverage_ratio = (semantic_coverage * 0.6 + normalized_depth * 0.4) if all_texts else 0
            
            final_score = (
                coverage_ratio * 0.4 +
                domain_variety_score * 0.3 +
                cross_referencing_score * 0.3
            )
            
            evaluation_result = {
                'coverage_ratio': coverage_ratio,
                'semantic_coverage': semantic_coverage,
                'missed_relevant_sources': [],
                'total_sources': len(all_sources),
                'unique_domains': len(unique_domains),
                'source_depth': source_depth,
                'cross_referencing_score': cross_referencing_score,
                'domain_variety_score': domain_variety_score
            }

            if trace_id and self.langfuse_tracker:
                await self.langfuse_tracker.track_coverage_metric(
                    trace_id, final_score, evaluation_result
                )

            return final_score, evaluation_result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in source coverage evaluation: {error_msg}", exc_info=True)
            if trace_id and self.langfuse_tracker:
                self.langfuse_tracker.track_error(trace_id, error_msg)
            return 0.0, {'error': error_msg}

def create_source_coverage_evaluator(
    langfuse_public_key: Optional[str] = None,
    langfuse_secret_key: Optional[str] = None,
    langfuse_host: Optional[str] = None
) -> SourceCoverageEvaluator:
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
    
    return SourceCoverageEvaluator(langfuse_tracker=langfuse_tracker)