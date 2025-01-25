import logging
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.research.common.model_schemas import ResearchToolOutput
from typing import List, Dict, Tuple
import json
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

nlp = spacy.blank('en')
nlp.add_pipe('sentencizer')

class AnswerRelevanceEvaluator:
    def __init__(self):
        self.nlp = nlp
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    def calculate_similarity(self, text1, text2):
        vectors = self.vectorizer.fit_transform([text1, text2])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    def evaluate_answer_relevance(self, research_output: ResearchToolOutput, query: str) -> Tuple[float, Dict]:
        all_sources = [
            {
                'text': content.content or content.snippet, 
                'url': content.url
            } for content in research_output.content
        ]
        
        full_text = research_output.summary or " ".join(content.content for content in research_output.content)
        
        if not full_text.strip():
            return 0.0, {'message': 'No valid text to evaluate'}
        
        similarity = self.calculate_similarity(query, full_text)
        
        query_doc = self.nlp(query)
        answer_doc = self.nlp(full_text)
        
        query_keywords = set(token.lemma_.lower() for token in query_doc if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop)
        answer_keywords = set(token.lemma_.lower() for token in answer_doc if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop)
        
        keyword_overlap = len(query_keywords.intersection(answer_keywords)) / len(query_keywords) if query_keywords else 0
        
        # Additional metrics
        entity_coverage = 0  # Calculate based on named entities
        topic_focus = similarity  # Can be more sophisticated
        off_topic_sentences = []  # Identify sentences less relevant to query
        
        score = (similarity + keyword_overlap) / 2
        
        return score, {
            'similarity': similarity,
            'keyword_overlap': keyword_overlap,
            'entity_coverage': entity_coverage,
            'topic_focus': topic_focus,
            'off_topic_sentences': off_topic_sentences,
            'total_sources': len(all_sources),
            'query_match_percentage': score * 100
        }

def create_answer_relevance_evaluator():
    return AnswerRelevanceEvaluator()