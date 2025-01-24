import spacy
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.research.common.model_schemas import ResearchToolOutput

nlp = spacy.blank('en')
nlp.add_pipe('sentencizer')

class LogicalCoherenceEvaluator:
    def __init__(self):
        self.nlp = nlp
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )

    def calculate_sentence_similarity(self, sentences: List[str]) -> List[float]:
        if len(sentences) < 2:
            return []
        vectors = self.vectorizer.fit_transform(sentences)
        return [cosine_similarity(vectors[i:i+1], vectors[i+1:i+2])[0][0] for i in range(len(sentences) - 1)]

    def evaluate_logical_coherence(self, research_output: ResearchToolOutput) -> Tuple[float, Dict]:
        full_text = research_output.summary or " ".join(content.content for content in research_output.content)
        
        if not full_text.strip():  # Check if the full text is empty
            return 0.0, {'message': 'No valid text to evaluate'}
        
        doc = self.nlp(full_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]  # Remove empty sentences
        
        if len(sentences) < 2:  # Ensure there are enough sentences to evaluate
            return 0.0, {'message': 'Not enough sentences to evaluate coherence'}
        
        transition_scores = self.calculate_sentence_similarity(sentences)
        rough_transitions = [{'sentence1': sentences[i], 'sentence2': sentences[i + 1], 'score': score} 
                             for i, score in enumerate(transition_scores) if score < 0.3]
        
        semantic_connection_scores = [cosine_similarity(self.vectorizer.transform([sentences[i]]), 
                                      self.vectorizer.transform([sentences[i+1]]))[0][0] 
                                      for i in range(len(sentences) - 1)]
        semantic_connection_score = np.mean(semantic_connection_scores) if semantic_connection_scores else 0
        
        idea_progression_scores = [1 / (1 + abs(len(sentences[i+1].split()) - len(sentences[i].split()))) 
                                   for i in range(len(sentences) - 1)]
        idea_progression_score = np.mean(idea_progression_scores) if idea_progression_scores else 0
        
        logical_fallacies_indicators = ['however', 'but', 'nevertheless', 'despite', 'although',
                                        'seems', 'appears', 'might', 'could', 'possibly']
        logical_fallacies_count = sum(1 for sent in sentences for indicator in logical_fallacies_indicators 
                                      if indicator in sent.lower())
        
        flow_score, has_argument_structure, has_discourse_markers, paragraph_score, topic_coherence = 0.5, False, False, 0.5, 0.5
        
        final_score = max(0.0, min(1.0, flow_score * 0.3 + float(has_argument_structure) * 0.2 + 
                                   float(has_discourse_markers) * 0.1 + paragraph_score * 0.2 + topic_coherence * 0.2))
        
        return final_score, {
            'flow_score': flow_score,
            'has_argument_structure': has_argument_structure,
            'has_discourse_markers': has_discourse_markers,
            'paragraph_score': paragraph_score,
            'rough_transitions': rough_transitions,
            'total_sentences': len(sentences),
            'semantic_connection_score': semantic_connection_score,
            'idea_progression_score': idea_progression_score,
            'logical_fallacies_count': logical_fallacies_count
        }

def create_logical_coherence_evaluator() -> LogicalCoherenceEvaluator:
    return LogicalCoherenceEvaluator()
