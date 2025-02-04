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
        all_sources = [{'url': content.url, 'text': content.content or content.snippet} 
                    for content in research_output.content]
        unique_domains = set(urllib.parse.urlparse(source['url']).netloc 
                            for source in all_sources if source['url'])
        
        def calculate_source_depth(url):
            return len(urllib.parse.urlparse(url).path.strip('/').split('/'))
        
        # Calculate source depths
        source_depths = [calculate_source_depth(source['url']) for source in all_sources]
        source_depth = np.mean(source_depths) if source_depths else 0
        
        # Calculate semantic coverage and cross-referencing
        all_texts = [source['text'] for source in all_sources if source['text']]
        if all_texts:
            self.vectorizer.fit(all_texts)
            vectors = self.vectorizer.transform(all_texts)
            
            # Calculate semantic coverage matrix
            semantic_similarity_matrix = cosine_similarity(vectors)
            
            # Calculate average semantic coverage
            # Exclude self-similarities (diagonal) and take upper triangle
            mask = np.triu(np.ones_like(semantic_similarity_matrix), k=1)
            semantic_coverage = np.mean(semantic_similarity_matrix[mask > 0]) if len(all_texts) > 1 else 0
            
            # Calculate cross-referencing
            pairs = list(combinations(range(len(all_texts)), 2))
            cross_references = sum(semantic_similarity_matrix[i][j] > 0.3 for i, j in pairs)
            cross_referencing_score = cross_references / len(pairs) if pairs else 0
        else:
            semantic_coverage = 0
            cross_referencing_score = 0
        
        # Calculate domain variety
        domain_variety_score = len(unique_domains) / len(all_sources) if all_sources else 0
        
        # Calculate coverage ratio based on semantic coverage and source depth
        normalized_depth = min(source_depth / 5, 1.0)  # Normalize depth with max value of 5
        coverage_ratio = (semantic_coverage * 0.6 + normalized_depth * 0.4) if all_texts else 0
        
        # Calculate final score with weighted components
        final_score = (
            coverage_ratio * 0.4 +          # Coverage ratio (semantic + depth)
            domain_variety_score * 0.3 +    # Domain variety
            cross_referencing_score * 0.3   # Cross-referencing
        )
        
        return final_score, {
            'coverage_ratio': coverage_ratio,
            'semantic_coverage': semantic_coverage,  # Changed from diversity_score
            'missed_relevant_sources': [],
            'total_sources': len(all_sources),
            'unique_domains': len(unique_domains),
            'source_depth': source_depth,
            'cross_referencing_score': cross_referencing_score,
            'domain_variety_score': domain_variety_score
        }

def create_source_coverage_evaluator() -> SourceCoverageEvaluator:
    return SourceCoverageEvaluator()
