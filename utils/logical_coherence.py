import spacy
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.research.common.model_schemas import ResearchToolOutput
from langfuse import Langfuse
from concurrent.futures import ThreadPoolExecutor
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LangfuseTracker:
   def __init__(self, 
                public_key: Optional[str] = None, 
                secret_key: Optional[str] = None, 
                host: Optional[str] = None):
       # Use environment variables if not explicitly provided
       public_key = public_key or os.getenv('LANGFUSE_PUBLIC_KEY')
       secret_key = secret_key or os.getenv('LANGFUSE_SECRET_KEY')
       host = host or os.getenv('LANGFUSE_HOST')

       if not public_key or not secret_key:
           raise ValueError("Langfuse public and secret keys must be provided")

       self.langfuse = Langfuse(
           public_key=public_key, 
           secret_key=secret_key, 
           host=host
       )
       self._executor = ThreadPoolExecutor(max_workers=4)

    async def track_coherence_metric(self, trace_id: str, score: float, details: Dict) -> None:
        try:
            generation = self.langfuse.get_generation(trace_id)
            generation.score(
                name="logical-coherence",
                value=score,
                metadata={
                    'topic_coherence': details.get('topic_coherence', 0.0),
                    'flow_score': details.get('flow_score', 0.0),
                    'structure_score': details.get('structure_score', 0.0),
                    'discourse_score': details.get('discourse_score', 0.0),
                    'has_argument_structure': details.get('has_argument_structure', False),
                    'rough_transitions_count': len(details.get('rough_transitions', [])),
                    'total_sentences': details.get('total_sentences', 0),
                    'timestamp': datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error tracking coherence metrics: {str(e)}", exc_info=True)

    def track_error(self, trace_id: str, error: str) -> None:
        try:
            trace = self.langfuse.get_trace(trace_id)
            trace.error(
                error=error,
                metadata={
                    "component": "logical_coherence",
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error tracking error event: {str(e)}", exc_info=True)

class LogicalCoherenceEvaluator:
    def __init__(self, langfuse_tracker: Optional[LangfuseTracker] = None):
        self.nlp = spacy.blank('en')
        self.nlp.add_pipe('sentencizer')
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.discourse_markers = {
            'addition': ['furthermore', 'moreover', 'additionally', 'also', 'besides'],
            'contrast': ['however', 'nevertheless', 'although', 'conversely', 'whereas'],
            'cause_effect': ['therefore', 'thus', 'consequently', 'because', 'since'],
            'sequence': ['first', 'second', 'finally', 'next', 'subsequently'],
            'example': ['for example', 'for instance', 'specifically', 'namely'],
            'summary': ['in conclusion', 'to summarize', 'in summary', 'overall']
        }
        self.langfuse_tracker = langfuse_tracker
        logger.info("Initialized LogicalCoherenceEvaluator")

    async def calculate_topic_coherence(self, sentences: List[str]) -> float:
        if len(sentences) < 2:
            return 0.0
        try:
            vectors = self.vectorizer.fit_transform(sentences)
            similarities = []
            for i in range(len(sentences) - 1):
                similarity = cosine_similarity(vectors[i:i+1], vectors[i+1:i+2])[0][0]
                similarities.append(similarity)
            return float(np.mean(similarities)) if similarities else 0.0
        except Exception as e:
            logger.error(f"Error calculating topic coherence: {str(e)}", exc_info=True)
            return 0.0

    def analyze_discourse_markers(self, text: str) -> Dict[str, List[str]]:
        found_markers = {category: [] for category in self.discourse_markers}
        for category, markers in self.discourse_markers.items():
            for marker in markers:
                if marker in text.lower():
                    found_markers[category].append(marker)
        return found_markers

    def calculate_argument_structure(self, sentences: List[str]) -> Tuple[bool, float]:
        has_intro = any('introduction' in sent.lower() for sent in sentences[:3])
        has_conclusion = any('conclusion' in sent.lower() for sent in sentences[-3:])
        argument_indicators = ['because', 'therefore', 'thus', 'since', 'consequently']
        has_arguments = any(indicator in ' '.join(sentences).lower() 
                          for indicator in argument_indicators)
        structure_score = (
            0.3 * float(has_intro) + 
            0.3 * float(has_conclusion) + 
            0.4 * float(has_arguments)
        )
        return has_arguments, structure_score

    async def evaluate_logical_coherence(
        self, 
        research_output: ResearchToolOutput,
        trace_id: Optional[str] = None
    ) -> Tuple[float, Dict]:
        try:
            full_text = research_output.summary or " ".join(
                content.content for content in research_output.content
            )
            
            if not full_text.strip():
                return 0.0, {'error': 'No valid text to evaluate'}
                
            doc = self.nlp(full_text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
            if len(sentences) < 2:
                return 0.0, {'error': 'Not enough sentences to evaluate coherence'}
                
            topic_coherence = await self.calculate_topic_coherence(sentences)
            discourse_markers = self.analyze_discourse_markers(full_text)
            has_argument_structure, structure_score = self.calculate_argument_structure(sentences)
            
            transition_scores = [
                cosine_similarity(
                    self.vectorizer.fit_transform([sentences[i]]),
                    self.vectorizer.fit_transform([sentences[i+1]])
                )[0][0]
                for i in range(len(sentences) - 1)
            ]
            
            flow_score = float(np.mean(transition_scores)) if transition_scores else 0.0
            
            rough_transitions = [
                {
                    'sentence1': sentences[i],
                    'sentence2': sentences[i + 1],
                    'score': score
                }
                for i, score in enumerate(transition_scores)
                if score < 0.3
            ]
            
            discourse_score = min(1.0, len([m for ms in discourse_markers.values() 
                                       for m in ms]) / len(sentences))
                                       
            final_score = (
                0.3 * topic_coherence +
                0.25 * flow_score +
                0.25 * structure_score +
                0.2 * discourse_score
            )
            
            final_score = max(0.0, min(1.0, final_score))
            
            evaluation_result = {
                'topic_coherence': topic_coherence,
                'flow_score': flow_score,
                'structure_score': structure_score,
                'discourse_score': discourse_score,
                'has_argument_structure': has_argument_structure,
                'discourse_markers': discourse_markers,
                'rough_transitions': rough_transitions,
                'total_sentences': len(sentences)
            }

            if trace_id and self.langfuse_tracker:
                await self.langfuse_tracker.track_coherence_metric(
                    trace_id, final_score, evaluation_result
                )

            return final_score, evaluation_result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in coherence evaluation: {error_msg}", exc_info=True)
            if trace_id and self.langfuse_tracker:
                self.langfuse_tracker.track_error(trace_id, error_msg)
            return 0.0, {'error': error_msg}

def create_logical_coherence_evaluator(
   langfuse_public_key: Optional[str] = None,
   langfuse_secret_key: Optional[str] = None,
   langfuse_host: Optional[str] = None
) -> LogicalCoherenceEvaluator:
   # Use environment variables if not explicitly provided
   langfuse_public_key = langfuse_public_key or os.getenv('LANGFUSE_PUBLIC_KEY')
   langfuse_secret_key = langfuse_secret_key or os.getenv('LANGFUSE_SECRET_KEY')
   langfuse_host = langfuse_host or os.getenv('LANGFUSE_HOST')

   langfuse_tracker = None
   if langfuse_public_key and langfuse_secret_key:
       try:
           langfuse_tracker = LangfuseTracker(
               public_key=langfuse_public_key,
               secret_key=langfuse_secret_key,
               host=langfuse_host
           )
           logger.info("Langfuse integration enabled")
       except Exception as e:
           logger.error(f"Failed to initialize Langfuse: {str(e)}")
   
   return LogicalCoherenceEvaluator(langfuse_tracker=langfuse_tracker)