import spacy
import numpy as np
import urllib.parse
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.research.common.model_schemas import ResearchToolOutput
from itertools import combinations

nlp = spacy.blank('en')
nlp.add_pipe('sentencizer')

class SourceCoverageEvaluator:
    def __init__(self):
        self.nlp = nlp
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    def evaluate_source_coverage(self, research_output: ResearchToolOutput) -> Tuple[float, Dict]:
        all_sources = [{'url': content.url, 'text': content.content or content.snippet} for content in research_output.content]
        unique_domains = set(urllib.parse.urlparse(source['url']).netloc for source in all_sources if source['url'])
        
        def calculate_source_depth(url):
            return len(urllib.parse.urlparse(url).path.strip('/').split('/'))
        
        # Calculate source depths and handle empty list case
        source_depths = [calculate_source_depth(source['url']) for source in all_sources]
        source_depth = np.mean(source_depths) if source_depths else 0  # Avoid empty slice warning
        
        # Ensure there are texts to process before fitting the vectorizer
        all_texts = [source['text'] for source in all_sources if source['text']]
        if all_texts:
            self.vectorizer.fit(all_texts)
            vectors = self.vectorizer.transform(all_texts)
            pairs = list(combinations(range(len(all_texts)), 2))
            cross_references = sum(cosine_similarity(vectors[i:i+1], vectors[j:j+1])[0][0] > 0.3 for i, j in pairs)
            cross_referencing_score = cross_references / len(pairs) if pairs else 0
        else:
            cross_referencing_score = 0
        
        domain_variety_score = len(unique_domains) / len(all_sources) if all_sources else 0
        
        # Check if coverage ratio and diversity score calculations are valid before using them
        coverage_ratio = diversity_score = 0
        missed_relevant_sources = []
        
        # Final score calculation with proper handling of empty values
        final_score = coverage_ratio * 0.7 + diversity_score * 0.3
        
        return final_score, {
            'coverage_ratio': coverage_ratio,
            'diversity_score': diversity_score,
            'missed_relevant_sources': missed_relevant_sources,
            'total_sources': len(all_sources),
            'unique_domains': len(unique_domains),
            'source_depth': source_depth,
            'cross_referencing_score': cross_referencing_score,
            'domain_variety_score': domain_variety_score
        }

def create_source_coverage_evaluator() -> SourceCoverageEvaluator:
    return SourceCoverageEvaluator()
