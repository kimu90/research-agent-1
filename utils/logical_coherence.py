import spacy
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.research.common.model_schemas import ResearchToolOutput

class LogicalCoherenceEvaluator:
    def __init__(self):
        # Initialize spacy with blank model and add sentencizer
        self.nlp = spacy.blank('en')
        self.nlp.add_pipe('sentencizer')
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams for better topic detection
        )
        
        # Define discourse markers for coherence analysis
        self.discourse_markers = {
            'addition': ['furthermore', 'moreover', 'additionally', 'also', 'besides'],
            'contrast': ['however', 'nevertheless', 'although', 'conversely', 'whereas'],
            'cause_effect': ['therefore', 'thus', 'consequently', 'because', 'since'],
            'sequence': ['first', 'second', 'finally', 'next', 'subsequently'],
            'example': ['for example', 'for instance', 'specifically', 'namely'],
            'summary': ['in conclusion', 'to summarize', 'in summary', 'overall']
        }

    def calculate_topic_coherence(self, sentences: List[str]) -> float:
        """
        Calculate topic coherence using TF-IDF and cosine similarity
        """
        if len(sentences) < 2:
            return 0.0
            
        try:
            # Create document-term matrix
            vectors = self.vectorizer.fit_transform(sentences)
            
            # Calculate pairwise similarities between consecutive sentences
            similarities = []
            for i in range(len(sentences) - 1):
                similarity = cosine_similarity(
                    vectors[i:i+1], 
                    vectors[i+1:i+2]
                )[0][0]
                similarities.append(similarity)
            
            # Return average similarity as topic coherence score
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception as e:
            print(f"Error calculating topic coherence: {str(e)}")
            return 0.0

    def analyze_discourse_markers(self, text: str) -> Dict[str, List[str]]:
        """
        Analyze discourse markers present in the text
        """
        found_markers = {category: [] for category in self.discourse_markers}
        
        for category, markers in self.discourse_markers.items():
            for marker in markers:
                if marker in text.lower():
                    found_markers[category].append(marker)
                    
        return found_markers

    def calculate_argument_structure(self, sentences: List[str]) -> Tuple[bool, float]:
        """
        Evaluate argument structure and calculate a structure score
        """
        has_intro = any('introduction' in sent.lower() for sent in sentences[:3])
        has_conclusion = any('conclusion' in sent.lower() for sent in sentences[-3:])
        
        # Look for argument indicators
        argument_indicators = ['because', 'therefore', 'thus', 'since', 'consequently']
        has_arguments = any(indicator in ' '.join(sentences).lower() 
                          for indicator in argument_indicators)
        
        # Calculate structure score
        structure_score = (0.3 * float(has_intro) + 
                         0.3 * float(has_conclusion) + 
                         0.4 * float(has_arguments))
                         
        return has_arguments, structure_score

    def evaluate_logical_coherence(self, research_output: ResearchToolOutput) -> Tuple[float, Dict]:
        """
        Evaluate logical coherence of research output
        """
        # Extract and validate text
        full_text = research_output.summary or " ".join(
            content.content for content in research_output.content
        )
        
        if not full_text.strip():
            return 0.0, {'error': 'No valid text to evaluate'}
            
        # Process text into sentences
        doc = self.nlp(full_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        if len(sentences) < 2:
            return 0.0, {'error': 'Not enough sentences to evaluate coherence'}
            
        # Calculate core metrics
        topic_coherence = self.calculate_topic_coherence(sentences)
        discourse_markers = self.analyze_discourse_markers(full_text)
        has_argument_structure, structure_score = self.calculate_argument_structure(sentences)
        
        # Calculate sentence-level flow
        transition_scores = [
            cosine_similarity(
                self.vectorizer.fit_transform([sentences[i]]),
                self.vectorizer.fit_transform([sentences[i+1]])
            )[0][0]
            for i in range(len(sentences) - 1)
        ]
        
        flow_score = float(np.mean(transition_scores)) if transition_scores else 0.0
        
        # Identify rough transitions
        rough_transitions = [
            {
                'sentence1': sentences[i],
                'sentence2': sentences[i + 1],
                'score': score
            }
            for i, score in enumerate(transition_scores)
            if score < 0.3
        ]
        
        # Calculate final coherence score
        weights = {
            'topic_coherence': 0.3,
            'flow': 0.25,
            'structure': 0.25,
            'discourse': 0.2
        }
        
        discourse_score = min(1.0, len([m for ms in discourse_markers.values() 
                                      for m in ms]) / len(sentences))
                                      
        final_score = (
            weights['topic_coherence'] * topic_coherence +
            weights['flow'] * flow_score +
            weights['structure'] * structure_score +
            weights['discourse'] * discourse_score
        )
        
        # Ensure score is in valid range
        final_score = max(0.0, min(1.0, final_score))
        
        return final_score, {
            'topic_coherence': topic_coherence,
            'flow_score': flow_score,
            'structure_score': structure_score,
            'discourse_score': discourse_score,
            'has_argument_structure': has_argument_structure,
            'discourse_markers': discourse_markers,
            'rough_transitions': rough_transitions,
            'total_sentences': len(sentences)
        }

def create_logical_coherence_evaluator() -> LogicalCoherenceEvaluator:
    """Factory function to create a LogicalCoherenceEvaluator instance"""
    return LogicalCoherenceEvaluator()