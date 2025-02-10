from typing import Dict, Any, Tuple, List, Optional
import re
import logging
import asyncio
from datetime import datetime
from langfuse import Langfuse
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)

class LangfuseTracker:
    def __init__(self, public_key: str, secret_key: str, host: Optional[str] = None):
        self.langfuse = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        self._executor = ThreadPoolExecutor(max_workers=4)

    async def track_test_metric(self, trace_id: str, score: float, details: Dict[str, Any]) -> None:
        try:
            generation = self.langfuse.get_generation(trace_id)
            generation.score(
                name="automated-test",
                value=score,
                metadata={
                    'rouge_scores': details.get('rouge_scores', {}),
                    'semantic_similarity': details.get('semantic_similarity', 0.0),
                    'hallucination_score': details.get('hallucination_score', 0.0),
                    'suspicious_segments_count': len(details.get('suspicious_segments', [])),
                    'timestamp': datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error tracking test metrics: {str(e)}", exc_info=True)

    def track_error(self, trace_id: str, error: str) -> None:
        try:
            trace = self.langfuse.get_trace(trace_id)
            trace.error(
                error=error,
                metadata={
                    "component": "automated_test",
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error tracking error event: {str(e)}", exc_info=True)

class AutomatedTestEvaluator:
    def __init__(self, langfuse_tracker: Optional[LangfuseTracker] = None):
        self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        self.langfuse_tracker = langfuse_tracker
        logger.info("Initialized AutomatedTestEvaluator")

    def preprocess(self, text: str) -> List[str]:
        text = text.lower()
        words = re.findall(r'\w+', text)
        return [w for w in words if w not in self.stopwords]

    def calculate_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        try:
            pred_words = self.preprocess(prediction)
            ref_words = self.preprocess(reference)
            
            # ROUGE-1
            common_unigrams = set(pred_words) & set(ref_words)
            rouge1 = len(common_unigrams) / max(len(set(pred_words)), 1)
            
            # ROUGE-2
            pred_bigrams = set(zip(pred_words[:-1], pred_words[1:]))
            ref_bigrams = set(zip(ref_words[:-1], ref_words[1:]))
            common_bigrams = pred_bigrams & ref_bigrams
            rouge2 = len(common_bigrams) / max(len(pred_bigrams), 1)
            
            return {
                'rouge1': rouge1,
                'rouge2': rouge2,
                'rougeL': rouge1
            }
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {str(e)}", exc_info=True)
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}

    def calculate_semantic_similarity(self, prediction: str, reference: str) -> float:
        try:
            pred_words = self.preprocess(prediction)
            ref_words = self.preprocess(reference)
            
            pred_set = set(pred_words)
            ref_set = set(ref_words)
            
            intersection = len(pred_set & ref_set)
            union = len(pred_set | ref_set)
            
            return intersection / max(union, 1)
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}", exc_info=True)
            return 0.0

    def detect_hallucination(self, prediction: str, reference: str) -> Tuple[float, List[str]]:
        try:
            pred_sentences = [s.strip() for s in prediction.split('.') if s.strip()]
            ref_sentences = [s.strip() for s in reference.split('.') if s.strip()]
            
            suspicious_segments = []
            hallucination_score = 0
            
            for pred_sent in pred_sentences:
                max_similarity = 0
                pred_words = set(self.preprocess(pred_sent))
                
                for ref_sent in ref_sentences:
                    ref_words = set(self.preprocess(ref_sent))
                    similarity = len(pred_words & ref_words) / max(len(pred_words | ref_words), 1)
                    max_similarity = max(max_similarity, similarity)
                
                if max_similarity < 0.3:
                    suspicious_segments.append(pred_sent)
                    hallucination_score += 1
            
            hallucination_score = hallucination_score / max(len(pred_sentences), 1)
            return hallucination_score, suspicious_segments
        except Exception as e:
            logger.error(f"Error detecting hallucinations: {str(e)}", exc_info=True)
            return 1.0, []

    async def evaluate_automated_tests(
        self, 
        prediction: str, 
        reference: str,
        trace_id: Optional[str] = None
    ) -> Tuple[float, Dict[str, Any]]:
        try:
            logger.info("Starting automated tests evaluation")
            
            rouge_scores = self.calculate_rouge_scores(prediction, reference)
            semantic_sim = self.calculate_semantic_similarity(prediction, reference)
            hallucination_score, suspicious_segments = self.detect_hallucination(prediction, reference)

            overall_score = (
                rouge_scores['rouge1'] * 0.3 +
                rouge_scores['rouge2'] * 0.3 +
                semantic_sim * 0.2 +
                (1 - hallucination_score) * 0.2
            )

            details = {
                'rouge_scores': rouge_scores,
                'semantic_similarity': semantic_sim,
                'hallucination_score': hallucination_score,
                'suspicious_segments': suspicious_segments,
                'timestamp': datetime.now().isoformat()
            }

            if trace_id and self.langfuse_tracker:
                await self.langfuse_tracker.track_test_metric(
                    trace_id, overall_score, details
                )

            return overall_score, details

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in automated tests evaluation: {error_msg}", exc_info=True)
            if trace_id and self.langfuse_tracker:
                self.langfuse_tracker.track_error(trace_id, error_msg)
            return 0.0, {
                'error': error_msg,
                'rouge_scores': {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0},
                'semantic_similarity': 0.0,
                'hallucination_score': 1.0,
                'suspicious_segments': [],
                'timestamp': datetime.now().isoformat()
            }

def create_automated_test_evaluator(
    langfuse_public_key: Optional[str] = None,
    langfuse_secret_key: Optional[str] = None,
    langfuse_host: Optional[str] = None
) -> AutomatedTestEvaluator:
    langfuse_tracker = None
    if langfuse_public_key and langfuse_secret_key:
        try:
            langfuse_tracker = LangfuseTracker(
                public_key=langfuse_public_key,
                secret_key=langfuse_secret_key,
                host=langfuse_host
            )
            logger.info("Langfuse integration enabled")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {str(e)}")
    
    return AutomatedTestEvaluator(langfuse_tracker=langfuse_tracker)