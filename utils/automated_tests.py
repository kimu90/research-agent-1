from typing import Dict, Any, Tuple, List
import re
import logging
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)
from datetime import datetime

logger = logging.getLogger(__name__)

class AutomatedTestEvaluator:
    def __init__(self):
        self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def preprocess(self, text: str) -> List[str]:
        """Clean and tokenize text."""
        text = text.lower()
        words = re.findall(r'\w+', text)
        return [w for w in words if w not in self.stopwords]
    
    def calculate_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE-1, ROUGE-2 scores."""
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
            'rougeL': rouge1  # Simplified approximation
        }
    
    def calculate_semantic_similarity(self, prediction: str, reference: str) -> float:
        """Calculate semantic similarity using Jaccard similarity."""
        pred_words = self.preprocess(prediction)
        ref_words = self.preprocess(reference)
        
        pred_set = set(pred_words)
        ref_set = set(ref_words)
        
        intersection = len(pred_set & ref_set)
        union = len(pred_set | ref_set)
        
        return intersection / max(union, 1)
    
    def detect_hallucination(self, prediction: str, reference: str) -> Tuple[float, List[str]]:
        """Detect potential hallucinations in the prediction."""
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
    
    def evaluate_automated_tests(self, prediction: str, reference: str) -> Tuple[float, Dict[str, Any]]:
        """Run all automated tests and return aggregated results."""
        try:
            logger.info("Starting automated tests evaluation")
            logger.debug(f"Prediction length: {len(prediction)}")
            logger.debug(f"Reference length: {len(reference)}")

            rouge_scores = self.calculate_rouge_scores(prediction, reference)
            logger.info(f"ROUGE Scores: {rouge_scores}")

            semantic_sim = self.calculate_semantic_similarity(prediction, reference)
            logger.info(f"Semantic Similarity: {semantic_sim}")

            hallucination_score, suspicious_segments = self.detect_hallucination(prediction, reference)
            logger.info(f"Hallucination Score: {hallucination_score}")
            logger.info(f"Suspicious Segments: {suspicious_segments}")

            overall_score = (
                rouge_scores['rouge1'] * 0.3 +
                rouge_scores['rouge2'] * 0.3 +
                semantic_sim * 0.2 +
                (1 - hallucination_score) * 0.2
            )
            logger.info(f"Overall Score: {overall_score}")

            details = {
                'rouge_scores': rouge_scores,
                'semantic_similarity': semantic_sim,
                'hallucination_score': hallucination_score,
                'suspicious_segments': suspicious_segments,
                'timestamp': datetime.now().isoformat()
            }

            return overall_score, details

        except Exception as e:
            logger.error(f"Error in automated tests evaluation: {e}", exc_info=True)
            raise

def create_automated_test_evaluator():
    """Factory function to create AutomatedTestEvaluator instance."""
    return AutomatedTestEvaluator()