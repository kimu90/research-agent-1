from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import spacy
from tools.research.common.model_schemas import ResearchToolOutput, ContentItem
class SourceCoverageEvaluator:
    def __init__(self):
        """
        Initialize the Source Coverage Evaluator using TF-IDF
        """
        try:
            logging.info("Initializing TF-IDF vectorizer")
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english'
            )
            # Load small spacy model for basic text processing
            self.nlp = spacy.load('en_core_web_sm')
            logging.info("TF-IDF vectorizer initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize vectorizer: {e}")
            raise

    def evaluate_source_coverage(self, research_output: ResearchToolOutput) -> Tuple[float, Dict]:
        """
        Evaluate source coverage using TF-IDF vectors instead of transformers
        """
        logging.info("Starting source coverage evaluation")
        
        # Prepare sources and content
        all_sources = [
            {
                'url': content.url, 
                'text': content.content or content.snippet
            } for content in research_output.content
        ]
        
        # Get full text
        full_text = research_output.summary or " ".join(
            content.content for content in research_output.content
        )
        
        # Create document collection for vectorization
        documents = [full_text] + [source['text'] for source in all_sources]
        
        # Calculate TF-IDF vectors
        try:
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            # First vector is the full text, rest are sources
            response_vector = tfidf_matrix[0]
            source_vectors = tfidf_matrix[1:]
            
            # Calculate relevance scores
            relevance_scores = {}
            for idx, source in enumerate(all_sources):
                similarity = cosine_similarity(response_vector, source_vectors[idx])[0][0]
                relevance_scores[source['url']] = similarity
                
        except Exception as e:
            logging.error(f"Error in TF-IDF calculation: {e}")
            raise
            
        # Used sources
        used_source_urls = {content.url for content in research_output.content}
        
        # Find missed relevant sources (similarity threshold 0.3 for TF-IDF)
        missed_relevant_sources = []
        for source in all_sources:
            if source['url'] not in used_source_urls and relevance_scores[source['url']] > 0.3:
                missed_relevant_sources.append({
                    'url': source['url'],
                    'relevance_score': relevance_scores[source['url']]
                })
        
        # Calculate coverage metrics
        relevant_sources = [
            url for url, score in relevance_scores.items() 
            if score > 0.3
        ]
        
        coverage_ratio = (
            len(used_source_urls.intersection(relevant_sources)) / len(relevant_sources) 
            if relevant_sources else 0
        )
        
        # Calculate diversity using TF-IDF vectors
        used_source_vectors = [
            source_vectors[idx] for idx, source in enumerate(all_sources)
            if source['url'] in used_source_urls
        ]
        
        diversity_scores = []
        for i in range(len(used_source_vectors)):
            for j in range(i + 1, len(used_source_vectors)):
                similarity = cosine_similarity(used_source_vectors[i], used_source_vectors[j])[0][0]
                diversity_scores.append(similarity)
        
        diversity_score = 1 - (np.mean(diversity_scores) if diversity_scores else 0)
        
        # Combine metrics
        final_score = coverage_ratio * 0.7 + diversity_score * 0.3
        
        return final_score, {
            'coverage_ratio': coverage_ratio,
            'diversity_score': diversity_score,
            'missed_relevant_sources': missed_relevant_sources,
            'total_sources': len(all_sources)
        }

def create_source_coverage_evaluator() -> SourceCoverageEvaluator:
    """Factory function to create a SourceCoverageEvaluator"""
    return SourceCoverageEvaluator()