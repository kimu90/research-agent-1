import numpy as np
import spacy
import logging
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.research.common.model_schemas import ResearchToolOutput, ContentItem
from tools.research.common.model_schemas import ResearchToolOutput, ContentItem

class LogicalCoherenceEvaluator:
    def __init__(self, nlp_model: str = 'en_core_web_sm'):
        """
        Initialize the Logical Coherence Evaluator using TF-IDF and spaCy
        
        Args:
            nlp_model (str): spaCy NLP model name
        """
        try:
            logging.info("Initializing Logical Coherence Evaluator")
            self.nlp = spacy.load(nlp_model)
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english'
            )
            logging.info("Evaluation models loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            raise

    def calculate_sentence_similarity(self, sentences: List[str]) -> List[float]:
        """Helper method to calculate sentence similarities using TF-IDF"""
        if len(sentences) < 2:
            return []
            
        vectors = self.vectorizer.fit_transform(sentences)
        similarities = []
        
        for i in range(len(sentences) - 1):
            similarity = cosine_similarity(vectors[i:i+1], vectors[i+1:i+2])[0][0]
            similarities.append(similarity)
            
        return similarities

    def evaluate_logical_coherence(self, research_output: ResearchToolOutput) -> Tuple[float, Dict]:
        """
        Evaluate logical coherence using TF-IDF and linguistic features
        """
        # Prepare full text
        full_text = research_output.summary or " ".join(
            content.content for content in research_output.content
        )
        
        logging.info("Analyzing logical coherence")
        
        # Process text with spaCy
        doc = self.nlp(full_text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        # Calculate sentence-to-sentence coherence
        transition_scores = self.calculate_sentence_similarity(sentences)
        rough_transitions = []
        
        for i, score in enumerate(transition_scores):
            if score < 0.3:  # Adjusted threshold for TF-IDF
                rough_transitions.append({
                    'sentence1': sentences[i],
                    'sentence2': sentences[i + 1],
                    'score': score
                })
        
        # Calculate flow score
        flow_score = np.mean(transition_scores) if transition_scores else 0
        
        # Argument structure indicators
        arg_indicators = [
            'because', 'therefore', 'thus', 'consequently', 
            'however', 'although', 'moreover', 'furthermore'
        ]
        has_argument_structure = any(
            indicator in full_text.lower() 
            for indicator in arg_indicators
        )
        
        # Discourse markers
        discourse_markers = [
            'first', 'second', 'finally', 'in addition', 
            'consequently', 'furthermore', 'likewise'
        ]
        has_discourse_markers = any(
            marker in full_text.lower() 
            for marker in discourse_markers
        )
        
        # Paragraph structure analysis
        paragraphs = [p.strip() for p in full_text.split('\n\n') if p.strip()]
        paragraph_coherence = []
        
        for para in paragraphs:
            para_sentences = [sent.text.strip() for sent in self.nlp(para).sents]
            if len(para_sentences) > 1:
                similarities = self.calculate_sentence_similarity(para_sentences)
                if similarities:
                    paragraph_coherence.append(np.mean(similarities))
        
        # Calculate paragraph score
        paragraph_score = np.mean(paragraph_coherence) if paragraph_coherence else 0
        
        # Additional linguistic features
        sentence_lengths = [len(sent.split()) for sent in sentences]
        length_variance = np.var(sentence_lengths) if sentence_lengths else 0
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        
        # Topic consistency check using TF-IDF
        if len(sentences) > 1:
            topic_vectors = self.vectorizer.fit_transform(sentences)
            topic_coherence = np.mean([
                cosine_similarity(topic_vectors[0:1], topic_vectors[i:i+1])[0][0]
                for i in range(1, len(sentences))
            ])
        else:
            topic_coherence = 1.0
        
        # Combine metrics with additional features
        final_score = (
            flow_score * 0.3 + 
            float(has_argument_structure) * 0.2 + 
            float(has_discourse_markers) * 0.1 + 
            paragraph_score * 0.2 +
            topic_coherence * 0.2
        )
        
        # Normalize final score
        final_score = max(0.0, min(1.0, final_score))
        
        return final_score, {
            'flow_score': flow_score,
            'has_argument_structure': has_argument_structure,
            'has_discourse_markers': has_discourse_markers,
            'paragraph_score': paragraph_score,
            'rough_transitions': rough_transitions,
            'total_sentences': len(sentences),
            'total_paragraphs': len(paragraphs),
            'avg_sentence_length': avg_sentence_length,
            'length_variance': length_variance,
            'topic_coherence': topic_coherence
        }

def create_logical_coherence_evaluator() -> LogicalCoherenceEvaluator:
    """Factory function to create a LogicalCoherenceEvaluator"""
    return LogicalCoherenceEvaluator()