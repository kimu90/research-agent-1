import logging
from typing import Dict, Any, Optional
import numpy as np
import re
import sys
from datetime import datetime
from langfuse import Langfuse
from concurrent.futures import ThreadPoolExecutor
import asyncio

def setup_logging():
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(f'analysis_evaluator_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)

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

    async def track_analysis_metric(self, trace_id: str, component: str, 
                                  score: float, details: Dict[str, Any]) -> None:
        """Track individual analysis component metrics."""
        try:
            generation = self.langfuse.get_generation(trace_id)
            generation.score(
                name=f"analysis-{component.lower().replace(' ', '-')}",
                value=score,
                metadata=details
            )
        except Exception as e:
            logger.error(f"Error tracking analysis metric: {str(e)}", exc_info=True)

    def track_error(self, trace_id: str, error: str, component: str) -> None:
        """Track evaluation errors."""
        try:
            trace = self.langfuse.get_trace(trace_id)
            trace.error(
                error=error,
                metadata={
                    "component": component,
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error tracking error event: {str(e)}", exc_info=True)

class AnalysisEvaluator:
    def __init__(self, langfuse_tracker: Optional[LangfuseTracker] = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AnalysisEvaluator")
        self.langfuse_tracker = langfuse_tracker
        print("AnalysisEvaluator initialized and ready for evaluation")

    def evaluate_numerical_accuracy(self, result: Any, trace_id: Optional[str] = None) -> Dict[str, Any]:
        self.logger.debug("Starting numerical accuracy evaluation")
        print("\nEvaluating numerical accuracy...")
        
        try:
            numerical_pattern = r'\d+\.?\d*(?:\s*[+\-*/]\s*\d+\.?\d*)*\s*=\s*\d+\.?\d*'
            calculations = re.findall(numerical_pattern, result.analysis)
            
            self.logger.info(f"Found {len(calculations)} numerical calculations to evaluate")
            print(f"Found {len(calculations)} calculations to check")
            
            accuracy_score = 0.0
            correct_calcs = 0
            if calculations:
                for i, calc in enumerate(calculations, 1):
                    try:
                        left, right = calc.split('=')
                        expected = eval(left.strip())
                        actual = float(right.strip())
                        is_correct = abs(expected - actual) < 0.001
                        
                        if is_correct:
                            correct_calcs += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to evaluate calculation '{calc}': {str(e)}")
                        continue
                
                accuracy_score = correct_calcs / len(calculations)
            
            evaluation_result = {
                'score': accuracy_score,
                'details': {
                    'calculations_found': len(calculations),
                    'calculation_examples': calculations[:3],
                    'correct_calculations': correct_calcs
                }
            }

            if trace_id and self.langfuse_tracker:
                asyncio.create_task(
                    self.langfuse_tracker.track_analysis_metric(
                        trace_id, "numerical_accuracy", accuracy_score, evaluation_result['details']
                    )
                )

            return evaluation_result

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error in numerical accuracy evaluation: {error_msg}", exc_info=True)
            if trace_id and self.langfuse_tracker:
                self.langfuse_tracker.track_error(trace_id, error_msg, "numerical_accuracy")
            return {'score': 0.0, 'details': {'error': error_msg}}

    def evaluate_query_understanding(self, result: Any, query: str, 
                                   trace_id: Optional[str] = None) -> Dict[str, Any]:
        self.logger.debug("Starting query understanding evaluation")
        
        try:
            query_terms = set(query.lower().split())
            analysis_text = result.analysis.lower()
            covered_terms = sum(1 for term in query_terms if term in analysis_text)
            term_coverage = covered_terms / len(query_terms) if query_terms else 0
            
            analytical_elements = {
                'methodology': bool(re.search(r'method|approach|analysis|procedure', analysis_text)),
                'findings': bool(re.search(r'find|result|show|indicate|reveal', analysis_text)),
                'statistics': bool(re.search(r'statistics|average|median|correlation|distribution', analysis_text)),
                'data_reference': bool(re.search(r'data|dataset|sample|records|observations', analysis_text))
            }
            
            analytical_score = sum(analytical_elements.values()) / len(analytical_elements)
            total_score = (term_coverage + analytical_score) / 2
            
            evaluation_result = {
                'score': total_score,
                'details': {
                    'term_coverage': term_coverage,
                    **analytical_elements,
                    'covered_terms': covered_terms,
                    'total_query_terms': len(query_terms)
                }
            }

            if trace_id and self.langfuse_tracker:
                asyncio.create_task(
                    self.langfuse_tracker.track_analysis_metric(
                        trace_id, "query_understanding", total_score, evaluation_result['details']
                    )
                )

            return evaluation_result

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error in query understanding evaluation: {error_msg}", exc_info=True)
            if trace_id and self.langfuse_tracker:
                self.langfuse_tracker.track_error(trace_id, error_msg, "query_understanding")
            return {'score': 0.0, 'details': {'error': error_msg}}

    def evaluate_data_validation(self, result: Any, trace_id: Optional[str] = None) -> Dict[str, Any]:
        self.logger.debug("Starting data validation evaluation")
        
        try:
            analysis_text = result.analysis.lower()
            validation_checks = {
                'missing_data': bool(re.search(r'missing|null|empty|incomplete', analysis_text)),
                'outliers': bool(re.search(r'outlier|extreme|unusual|anomaly', analysis_text)),
                'distribution': bool(re.search(r'distribution|spread|range|variation', analysis_text)),
                'data_types': bool(re.search(r'type|format|numeric|categorical|string', analysis_text))
            }
            
            validation_score = sum(validation_checks.values()) / len(validation_checks)
            
            evaluation_result = {
                'score': validation_score,
                'details': validation_checks
            }

            if trace_id and self.langfuse_tracker:
                asyncio.create_task(
                    self.langfuse_tracker.track_analysis_metric(
                        trace_id, "data_validation", validation_score, validation_checks
                    )
                )

            return evaluation_result

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error in data validation evaluation: {error_msg}", exc_info=True)
            if trace_id and self.langfuse_tracker:
                self.langfuse_tracker.track_error(trace_id, error_msg, "data_validation")
            return {'score': 0.0, 'details': {'error': error_msg}}

    def evaluate_reasoning_transparency(self, result: Any, 
                                     trace_id: Optional[str] = None) -> Dict[str, Any]:
        self.logger.debug("Starting reasoning transparency evaluation")
        
        try:
            analysis_text = result.analysis.lower()
            sentences = self._simple_sentence_tokenize(analysis_text)
            
            transparency_checks = {
                'explains_steps': bool(re.search(r'first|then|next|finally|step|process', analysis_text)),
                'states_assumptions': bool(re.search(r'assume|assumption|given|consider|suppose', analysis_text)),
                'mentions_limitations': bool(re.search(r'limitation|caveat|constraint|restrict|note that', analysis_text)),
                'cites_evidence': bool(re.search(r'because|since|as shown|evidence|indicates', analysis_text)),
                'has_conclusion': bool(re.search(r'therefore|thus|conclude|in conclusion|overall', analysis_text)),
                'has_examples': bool(re.search(r'example|instance|case|specifically|such as', analysis_text))
            }
            
            transparency_score = sum(transparency_checks.values()) / len(transparency_checks)
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            
            evaluation_result = {
                'score': transparency_score,
                'details': {
                    **transparency_checks,
                    'avg_sentence_length': avg_sentence_length,
                    'total_sentences': len(sentences)
                }
            }

            if trace_id and self.langfuse_tracker:
                asyncio.create_task(
                    self.langfuse_tracker.track_analysis_metric(
                        trace_id, "reasoning_transparency", transparency_score, evaluation_result['details']
                    )
                )

            return evaluation_result

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error in reasoning transparency evaluation: {error_msg}", exc_info=True)
            if trace_id and self.langfuse_tracker:
                self.langfuse_tracker.track_error(trace_id, error_msg, "reasoning_transparency")
            return {'score': 0.0, 'details': {'error': error_msg}}

    def _simple_sentence_tokenize(self, text: str) -> list:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    async def evaluate_analysis(self, result: Any, query: str, 
                              trace_id: Optional[str] = None) -> Dict[str, Any]:
        self.logger.info("Starting comprehensive analysis evaluation")
        print("\n=== Starting Comprehensive Analysis Evaluation ===")
        
        try:
            numerical_accuracy = self.evaluate_numerical_accuracy(result, trace_id)
            query_understanding = self.evaluate_query_understanding(result, query, trace_id)
            data_validation = self.evaluate_data_validation(result, trace_id)
            reasoning_transparency = self.evaluate_reasoning_transparency(result, trace_id)
            
            component_scores = {
                'Numerical Accuracy': numerical_accuracy['score'],
                'Query Understanding': query_understanding['score'],
                'Data Validation': data_validation['score'],
                'Reasoning Transparency': reasoning_transparency['score']
            }
            
            overall_score = np.mean(list(component_scores.values()))
            
            evaluation_result = {
                'overall_score': overall_score,
                'numerical_accuracy': numerical_accuracy,
                'query_understanding': query_understanding,
                'data_validation': data_validation,
                'reasoning_transparency': reasoning_transparency
            }

            if trace_id and self.langfuse_tracker:
                await self.langfuse_tracker.track_analysis_metric(
                    trace_id, "overall", overall_score, {
                        'component_scores': component_scores,
                        'timestamp': datetime.now().isoformat()
                    }
                )

            return evaluation_result

        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"Error in overall analysis evaluation: {error_msg}", exc_info=True)
            if trace_id and self.langfuse_tracker:
                self.langfuse_tracker.track_error(trace_id, error_msg, "overall_analysis")
            return {'overall_score': 0.0, 'error': error_msg}

def create_analysis_evaluator(
   langfuse_public_key: Optional[str] = None,
   langfuse_secret_key: Optional[str] = None,
   langfuse_host: Optional[str] = None
) -> AnalysisEvaluator:
   """Create and return an instance of AnalysisEvaluator with optional Langfuse integration."""
   logger = logging.getLogger(__name__)
   logger.info("Creating new AnalysisEvaluator instance")
   
   # Use environment variables if not explicitly provided
   langfuse_public_key = langfuse_public_key or os.getenv('LANGFUSE_PUBLIC_KEY')
   langfuse_secret_key = langfuse_secret_key or os.getenv('LANGFUSE_SECRET_KEY')
   langfuse_host = langfuse_host or os.getenv('LANGFUSE_HOST')

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
   
   return AnalysisEvaluator(langfuse_tracker=langfuse_tracker)