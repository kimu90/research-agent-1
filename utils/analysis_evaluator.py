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
            has_findings = bool(re.search(r'find|result|show|indicate|reveal', analysis_text))
            has_statistics = bool(re.search(r'statistics|average|median|correlation|distribution', analysis_text))
            has_data_reference = bool(re.search(r'data|dataset|sample|records|observations', analysis_text))
            
            # Calculate comprehensive score
            analytical_elements = [has_methodology, has_findings, has_statistics, has_data_reference]
            analytical_score = sum(analytical_elements) / len(analytical_elements)
            
            # Combine scores
            total_score = (term_coverage + analytical_score) / 2
            
            return {
                'score': total_score,
                'details': {
                    'term_coverage': term_coverage,
                    'has_methodology': has_methodology,
                    'has_findings': has_findings,
                    'has_statistics': has_statistics,
                    'has_data_reference': has_data_reference,
                    'covered_terms': covered_terms,
                    'total_query_terms': len(query_terms)
                }
            }
        except Exception as e:
            self.logger.error(f"Error in query understanding evaluation: {str(e)}")
            return {'score': 0.0, 'details': {'error': str(e)}}

    def evaluate_data_validation(self, result: Any) -> Dict[str, Any]:
        """Evaluate the data validation aspects of the analysis."""
        try:
            analysis_text = result.analysis.lower()
            
            # Check for data quality mentions
            missing_data_check = bool(re.search(r'missing|null|empty|incomplete', analysis_text))
            outlier_check = bool(re.search(r'outlier|extreme|unusual|anomaly', analysis_text))
            distribution_check = bool(re.search(r'distribution|spread|range|variation', analysis_text))
            data_types_check = bool(re.search(r'type|format|numeric|categorical|string', analysis_text))
            
            # Check for validation steps
            validation_steps = [
                missing_data_check,
                outlier_check,
                distribution_check,
                data_types_check
            ]
            
            validation_score = sum(validation_steps) / len(validation_steps)
            
            return {
                'score': validation_score,
                'details': {
                    'missing_data_checked': missing_data_check,
                    'outliers_checked': outlier_check,
                    'distribution_checked': distribution_check,
                    'data_types_checked': data_types_check
                }
            }
        except Exception as e:
            self.logger.error(f"Error in data validation evaluation: {str(e)}")
            return {'score': 0.0, 'details': {'error': str(e)}}

    def evaluate_reasoning_transparency(self, result: Any) -> Dict[str, Any]:
        """Evaluate the transparency and clarity of the analytical reasoning."""
        try:
            analysis_text = result.analysis.lower()
            sentences = sent_tokenize(analysis_text)
            
            # Check for explanation patterns
            explains_steps = bool(re.search(r'first|then|next|finally|step|process', analysis_text))
            states_assumptions = bool(re.search(r'assume|assumption|given|consider|suppose', analysis_text))
            mentions_limitations = bool(re.search(r'limitation|caveat|constraint|restrict|note that', analysis_text))
            cites_evidence = bool(re.search(r'because|since|as shown|evidence|indicates', analysis_text))
            
            # Check for logical flow
            has_conclusion = bool(re.search(r'therefore|thus|conclude|in conclusion|overall', analysis_text))
            has_examples = bool(re.search(r'example|instance|case|specifically|such as', analysis_text))
            
            # Calculate transparency scores
            transparency_elements = [
                explains_steps,
                states_assumptions,
                mentions_limitations,
                cites_evidence,
                has_conclusion,
                has_examples
            ]
            
            transparency_score = sum(transparency_elements) / len(transparency_elements)
            
            # Analyze sentence structure
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            
            return {
                'score': transparency_score,
                'details': {
                    'explains_steps': explains_steps,
                    'states_assumptions': states_assumptions,
                    'mentions_limitations': mentions_limitations,
                    'cites_evidence': cites_evidence,
                    'has_conclusion': has_conclusion,
                    'has_examples': has_examples,
                    'avg_sentence_length': avg_sentence_length,
                    'total_sentences': len(sentences)
                }
            }
        except Exception as e:
            self.logger.error(f"Error in reasoning transparency evaluation: {str(e)}")
            return {'score': 0.0, 'details': {'error': str(e)}}

    def evaluate_analysis(self, result: Any, query: str) -> Dict[str, Any]:
        """Perform comprehensive evaluation of the analysis."""
        try:
            numerical_accuracy = self.evaluate_numerical_accuracy(result)
            query_understanding = self.evaluate_query_understanding(result, query)
            data_validation = self.evaluate_data_validation(result)
            reasoning_transparency = self.evaluate_reasoning_transparency(result)
            
            # Calculate overall score
            overall_score = np.mean([
                numerical_accuracy['score'],
                query_understanding['score'],
                data_validation['score'],
                reasoning_transparency['score']
            ])
            
            return {
                'overall_score': overall_score,
                'numerical_accuracy': numerical_accuracy,
                'query_understanding': query_understanding,
                'data_validation': data_validation,
                'reasoning_transparency': reasoning_transparency
            }
        except Exception as e:
            self.logger.error(f"Error in overall analysis evaluation: {str(e)}")
            return {
                'overall_score': 0.0,
                'error': str(e)
            }

def create_analysis_evaluator() -> AnalysisEvaluator:
    """Create and return an instance of AnalysisEvaluator."""
    return AnalysisEvaluator()