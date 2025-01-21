import numpy as np
import spacy
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.research.common.model_schemas import ResearchToolOutput, ContentItem
from tools.research.common.model_schemas import ResearchToolOutput, ContentItem

class AnswerRelevanceEvaluator:
    def __init__(self, nlp_model: str = 'en_core_web_sm'):
        """
        Initialize the Answer Relevance Evaluator using TF-IDF and spaCy
        """
        try:
            logging.info("Initializing Answer Relevance Evaluator")
            self.nlp = spacy.load(nlp_model)
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)  # Include bigrams for better context
            )
            logging.info("Evaluation models loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            raise

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using TF-IDF vectors"""
        vectors = self.vectorizer.fit_transform([text1, text2])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    def evaluate_answer_relevance(
        self, 
        research_output: ResearchToolOutput, 
        query: str
    ) -> Tuple[float, Dict]:
        """
        Evaluate relevance using TF-IDF and linguistic features
        """
        logging.info(f"Evaluating relevance for query: {query}")
        
        # Prepare full text
        full_text = research_output.summary or " ".join(
            content.content for content in research_output.content
        )
        
        # Calculate semantic similarity using TF-IDF
        semantic_similarity = self.calculate_text_similarity(query, full_text)
        
        # Extract key elements from query using spaCy
        query_doc = self.nlp(query)
        query_entities = set(ent.text.lower() for ent in query_doc.ents)
        query_keywords = set(
            token.lemma_.lower() 
            for token in query_doc 
            if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                not token.is_stop and 
                len(token.text) > 2)
        )
        
        # Process answer text
        answer_doc = self.nlp(full_text)
        answer_entities = set(ent.text.lower() for ent in answer_doc.ents)
        answer_keywords = set(
            token.lemma_.lower() 
            for token in answer_doc 
            if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                not token.is_stop and 
                len(token.text) > 2)
        )
        
        # Enhanced entity and keyword coverage
        entity_coverage = (
            len(query_entities.intersection(answer_entities)) / len(query_entities)
            if query_entities else 1
        )
        
        keyword_coverage = (
            len(query_keywords.intersection(answer_keywords)) / len(query_keywords)
            if query_keywords else 1
        )
        
        # Analyze sentence relevance
        sentences = [sent.text.strip() for sent in answer_doc.sents]
        sentence_scores = []
        off_topic_sentences = []
        
        for sent in sentences:
            # Calculate similarity between query and sentence
            score = self.calculate_text_similarity(query, sent)
            sentence_scores.append(score)
            
            # Identify off-topic sentences
            if score < 0.2:  # Adjusted threshold for TF-IDF
                off_topic_sentences.append({
                    'sentence': sent,
                    'relevance_score': score
                })
        
        # Calculate topic focus
        topic_focus = np.mean(sentence_scores) if sentence_scores else 0
        
        # Additional relevance features
        key_phrase_matches = self.analyze_key_phrases(query, full_text)
        context_relevance = self.analyze_context_relevance(query_doc, answer_doc)
        
        # Combine all metrics
        final_score = (
            semantic_similarity * 0.3 + 
            ((entity_coverage + keyword_coverage) / 2) * 0.3 + 
            topic_focus * 0.2 +
            key_phrase_matches * 0.1 +
            context_relevance * 0.1
        )
        
        # Normalize final score
        final_score = max(0.0, min(1.0, final_score))
        
        return final_score, {
            'semantic_similarity': semantic_similarity,
            'entity_coverage': entity_coverage,
            'keyword_coverage': keyword_coverage,
            'topic_focus': topic_focus,
            'key_phrase_matches': key_phrase_matches,
            'context_relevance': context_relevance,
            'off_topic_sentences': off_topic_sentences,
            'total_sentences': len(sentences)
        }

    def analyze_key_phrases(self, query: str, answer: str) -> float:
        """Analyze matching of key phrases between query and answer"""
        query_doc = self.nlp(query)
        answer_doc = self.nlp(answer)
        
        # Extract noun phrases and verb phrases
        query_phrases = set(chunk.text.lower() for chunk in query_doc.noun_chunks)
        answer_phrases = set(chunk.text.lower() for chunk in answer_doc.noun_chunks)
        
        # Calculate phrase overlap
        return (
            len(query_phrases.intersection(answer_phrases)) / len(query_phrases)
            if query_phrases else 1
        )

    def analyze_context_relevance(self, query_doc: spacy.tokens.Doc, answer_doc: spacy.tokens.Doc) -> float:
        """Analyze contextual relevance using dependency parsing"""
        # Get main verbs and their objects from query
        query_actions = set()
        for token in query_doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                # Get verb and its direct object if present
                obj = next((child.text for child in token.children if child.dep_ == "dobj"), "")
                query_actions.add((token.text, obj))
        
        # Look for similar patterns in answer
        matches = 0
        total_patterns = len(query_actions) if query_actions else 1
        
        for token in answer_doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                obj = next((child.text for child in token.children if child.dep_ == "dobj"), "")
                if any(token.similarity(self.nlp(q_verb)) > 0.7 for q_verb, _ in query_actions):
                    matches += 1
        
        return matches / total_patterns

def create_answer_relevance_evaluator() -> AnswerRelevanceEvaluator:
    """Factory function to create an AnswerRelevanceEvaluator"""
    return AnswerRelevanceEvaluator()