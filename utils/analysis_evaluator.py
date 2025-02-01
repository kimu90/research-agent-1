import logging
from typing import Dict, Any
import numpy as np
import re
import sys
from datetime import datetime

# Configure logging
def setup_logging():
    """Configure logging with both file and console handlers."""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create file handler
    file_handler = logging.FileHandler(f'analysis_evaluator_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Set up logging
logger = setup_logging()

# Suppress watchdog logging
logging.getLogger('watchdog.observers.inotify_buffer').setLevel(logging.WARNING)

class AnalysisEvaluator:
    """Evaluator for analysis outputs with comprehensive metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AnalysisEvaluator")
        print("AnalysisEvaluator initialized and ready for evaluation")

    def evaluate_numerical_accuracy(self, result: Any) -> Dict[str, Any]:
        """Evaluate the numerical accuracy of the analysis."""
        self.logger.debug("Starting numerical accuracy evaluation")
        print("\nEvaluating numerical accuracy...")
        
        try:
            numerical_pattern = r'\d+\.?\d*(?:\s*[+\-*/]\s*\d+\.?\d*)*\s*=\s*\d+\.?\d*'
            calculations = re.findall(numerical_pattern, result.analysis)
            
            self.logger.info(f"Found {len(calculations)} numerical calculations to evaluate")
            print(f"Found {len(calculations)} calculations to check")
            
            accuracy_score = 0.0
            if calculations:
                correct_calcs = 0
                for i, calc in enumerate(calculations, 1):
                    try:
                        left, right = calc.split('=')
                        expected = eval(left.strip())
                        actual = float(right.strip())
                        is_correct = abs(expected - actual) < 0.001
                        
                        self.logger.debug(f"Calculation {i}: {calc} - {'Correct' if is_correct else 'Incorrect'}")
                        if is_correct:
                            correct_calcs += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to evaluate calculation '{calc}': {str(e)}")
                        continue
                
                accuracy_score = correct_calcs / len(calculations)
                print(f"Accuracy score: {accuracy_score:.2%}")
            
            return {
                'score': accuracy_score,
                'details': {
                    'calculations_found': len(calculations),
                    'calculation_examples': calculations[:3]
                }
            }
        except Exception as e:
            self.logger.error(f"Error in numerical accuracy evaluation: {str(e)}", exc_info=True)
            print(f"Error in numerical evaluation: {str(e)}")
            return {'score': 0.0, 'details': {'error': str(e)}}

    def evaluate_query_understanding(self, result: Any, query: str) -> Dict[str, Any]:
        """Evaluate how well the analysis addresses the query."""
        self.logger.debug("Starting query understanding evaluation")
        print("\nEvaluating query understanding...")
        
        try:
            query_terms = set(query.lower().split())
            self.logger.info(f"Analyzing query with {len(query_terms)} unique terms")
            print(f"Query contains {len(query_terms)} unique terms")
            
            analysis_text = result.analysis.lower()
            covered_terms = sum(1 for term in query_terms if term in analysis_text)
            term_coverage = covered_terms / len(query_terms) if query_terms else 0
            
            self.logger.info(f"Term coverage: {term_coverage:.2%} ({covered_terms}/{len(query_terms)})")
            print(f"Terms covered: {covered_terms}/{len(query_terms)} ({term_coverage:.2%})")
            
            # Check for analytical elements
            analytical_elements = {
                'methodology': bool(re.search(r'method|approach|analysis|procedure', analysis_text)),
                'findings': bool(re.search(r'find|result|show|indicate|reveal', analysis_text)),
                'statistics': bool(re.search(r'statistics|average|median|correlation|distribution', analysis_text)),
                'data_reference': bool(re.search(r'data|dataset|sample|records|observations', analysis_text))
            }
            
            for element, present in analytical_elements.items():
                self.logger.debug(f"Analytical element '{element}': {'Present' if present else 'Missing'}")
            
            analytical_score = sum(analytical_elements.values()) / len(analytical_elements)
            total_score = (term_coverage + analytical_score) / 2
            
            print(f"Overall query understanding score: {total_score:.2%}")
            
            return {
                'score': total_score,
                'details': {
                    'term_coverage': term_coverage,
                    **analytical_elements,
                    'covered_terms': covered_terms,
                    'total_query_terms': len(query_terms)
                }
            }
        except Exception as e:
            self.logger.error(f"Error in query understanding evaluation: {str(e)}", exc_info=True)
            print(f"Error in query understanding evaluation: {str(e)}")
            return {'score': 0.0, 'details': {'error': str(e)}}

    def evaluate_data_validation(self, result: Any) -> Dict[str, Any]:
        """Evaluate the data validation aspects of the analysis."""
        self.logger.debug("Starting data validation evaluation")
        print("\nEvaluating data validation...")
        
        try:
            analysis_text = result.analysis.lower()
            
            validation_checks = {
                'missing_data': bool(re.search(r'missing|null|empty|incomplete', analysis_text)),
                'outliers': bool(re.search(r'outlier|extreme|unusual|anomaly', analysis_text)),
                'distribution': bool(re.search(r'distribution|spread|range|variation', analysis_text)),
                'data_types': bool(re.search(r'type|format|numeric|categorical|string', analysis_text))
            }
            
            for check, present in validation_checks.items():
                self.logger.debug(f"Validation check '{check}': {'Present' if present else 'Missing'}")
                print(f"- {check.replace('_', ' ').title()} check: {'✓' if present else '×'}")
            
            validation_score = sum(validation_checks.values()) / len(validation_checks)
            print(f"Overall validation score: {validation_score:.2%}")
            
            return {
                'score': validation_score,
                'details': validation_checks
            }
        except Exception as e:
            self.logger.error(f"Error in data validation evaluation: {str(e)}", exc_info=True)
            print(f"Error in data validation evaluation: {str(e)}")
            return {'score': 0.0, 'details': {'error': str(e)}}

    def evaluate_reasoning_transparency(self, result: Any) -> Dict[str, Any]:
        """Evaluate the transparency and clarity of the analytical reasoning."""
        self.logger.debug("Starting reasoning transparency evaluation")
        print("\nEvaluating reasoning transparency...")
        
        try:
            analysis_text = result.analysis.lower()
            # Use a simple sentence splitting method as an alternative to NLTK
            sentences = self._simple_sentence_tokenize(analysis_text)
            
            self.logger.info(f"Analyzing {len(sentences)} sentences for reasoning transparency")
            
            transparency_checks = {
                'explains_steps': bool(re.search(r'first|then|next|finally|step|process', analysis_text)),
                'states_assumptions': bool(re.search(r'assume|assumption|given|consider|suppose', analysis_text)),
                'mentions_limitations': bool(re.search(r'limitation|caveat|constraint|restrict|note that', analysis_text)),
                'cites_evidence': bool(re.search(r'because|since|as shown|evidence|indicates', analysis_text)),
                'has_conclusion': bool(re.search(r'therefore|thus|conclude|in conclusion|overall', analysis_text)),
                'has_examples': bool(re.search(r'example|instance|case|specifically|such as', analysis_text))
            }
            
            for check, present in transparency_checks.items():
                self.logger.debug(f"Transparency check '{check}': {'Present' if present else 'Missing'}")
                print(f"- {check.replace('_', ' ').title()}: {'✓' if present else '×'}")
            
            transparency_score = sum(transparency_checks.values()) / len(transparency_checks)
            
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            self.logger.info(f"Average sentence length: {avg_sentence_length:.1f} words")
            print(f"Average sentence length: {avg_sentence_length:.1f} words")
            
            return {
                'score': transparency_score,
                'details': {
                    **transparency_checks,
                    'avg_sentence_length': avg_sentence_length,
                    'total_sentences': len(sentences)
                }
            }
        except Exception as e:
            self.logger.error(f"Error in reasoning transparency evaluation: {str(e)}", exc_info=True)
            print(f"Error in reasoning transparency evaluation: {str(e)}")
            return {'score': 0.0, 'details': {'error': str(e)}}

    def _simple_sentence_tokenize(self, text: str) -> list:
        """
        A simple alternative to NLTK's sentence tokenization.
        Uses basic punctuation to split sentences.
        """
        # Split on periods, exclamation points, and question marks followed by spaces or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Remove any empty strings
        return [s.strip() for s in sentences if s.strip()]

    def evaluate_analysis(self, result: Any, query: str) -> Dict[str, Any]:
        """Perform comprehensive evaluation of the analysis."""
        self.logger.info("Starting comprehensive analysis evaluation")
        print("\n=== Starting Comprehensive Analysis Evaluation ===")
        
        try:
            numerical_accuracy = self.evaluate_numerical_accuracy(result)
            query_understanding = self.evaluate_query_understanding(result, query)
            data_validation = self.evaluate_data_validation(result)
            reasoning_transparency = self.evaluate_reasoning_transparency(result)
            
            component_scores = {
                'Numerical Accuracy': numerical_accuracy['score'],
                'Query Understanding': query_understanding['score'],
                'Data Validation': data_validation['score'],
                'Reasoning Transparency': reasoning_transparency['score']
            }
            
            for component, score in component_scores.items():
                self.logger.info(f"{component} score: {score:.2%}")
                print(f"{component} score: {score:.2%}")
            
            overall_score = np.mean(list(component_scores.values()))
            self.logger.info(f"Overall evaluation score: {overall_score:.2%}")
            print(f"\nOverall evaluation score: {overall_score:.2%}")
            
            return {
                'overall_score': overall_score,
                'numerical_accuracy': numerical_accuracy,
                'query_understanding': query_understanding,
                'data_validation': data_validation,
                'reasoning_transparency': reasoning_transparency
            }
        except Exception as e:
            self.logger.error(f"Error in overall analysis evaluation: {str(e)}", exc_info=True)
            print(f"Error in overall analysis evaluation: {str(e)}")
            return {
                'overall_score': 0.0,
                'error': str(e)
            }

def create_analysis_evaluator() -> AnalysisEvaluator:
    """Create and return an instance of AnalysisEvaluator."""
    logger = logging.getLogger(__name__)
    logger.info("Creating new AnalysisEvaluator instance")
    return AnalysisEvaluator()