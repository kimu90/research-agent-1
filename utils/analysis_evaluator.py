import logging
from typing import Dict, Any
import numpy as np
from scipy import stats
import re
from nltk.tokenize import sent_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class AnalysisEvaluator:
    """Evaluator for analysis outputs with comprehensive metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_numerical_accuracy(self, result: Any) -> Dict[str, Any]:
        """Evaluate the numerical accuracy of the analysis."""
        try:
            # Extract numerical statements and calculations
            numerical_pattern = r'\d+\.?\d*(?:\s*[+\-*/]\s*\d+\.?\d*)*\s*=\s*\d+\.?\d*'
            calculations = re.findall(numerical_pattern, result.analysis)
            
            accuracy_score = 0.0
            if calculations:
                correct_calcs = 0
                for calc in calculations:
                    try:
                        left, right = calc.split('=')
                        expected = eval(left.strip())
                        actual = float(right.strip())
                        if abs(expected - actual) < 0.001:  # Allow small floating point differences
                            correct_calcs += 1
                    except:
                        continue
                accuracy_score = correct_calcs / len(calculations) if calculations else 0.0
            
            return {
                'score': accuracy_score,
                'details': {
                    'calculations_found': len(calculations),
                    'calculation_examples': calculations[:3]  # Show first 3 examples
                }
            }
        except Exception as e:
            self.logger.error(f"Error in numerical accuracy evaluation: {str(e)}")
            return {'score': 0.0, 'details': {'error': str(e)}}

    def evaluate_query_understanding(self, result: Any, query: str) -> Dict[str, Any]:
        """Evaluate how well the analysis addresses the query."""
        try:
            # Extract key terms from query
            query_terms = set(query.lower().split())
            
            # Check analysis coverage
            analysis_text = result.analysis.lower()
            covered_terms = sum(1 for term in query_terms if term in analysis_text)
            term_coverage = covered_terms / len(query_terms) if query_terms else 0
            
            # Check for analytical elements
            has_methodology = bool(re.search(r'method|approach|analysis|procedure', analysis_text))
            has_findings = bool(re.search(r'find|result|show|indicate|reveal', analysis_