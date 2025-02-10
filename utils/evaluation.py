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
import os
from typing import Optional
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

    async def track_factual_metric(self, trace_id: str, score: float, details: Dict) -> None:
        try:
            generation = self.langfuse.get_generation(trace_id)
            generation.score(
                name="factual-accuracy",
                value=score,
                metadata={
                    'citation_accuracy': details.get('citation_accuracy', 0.0),
                    'total_sources': details.get('total_sources', 0),
                    'verified_claims': details.get('verified_claims', 0),
                    'unverified_claims': details.get('unverified_claims', 0),
                    'contradicting_claims': details.get('contradicting_claims', 0),
                    'source_credibility': details.get('source_credibility_score', 0.0),
                    'fact_check_coverage': details.get('fact_check_coverage', 0.0),
                    'timestamp': datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error tracking factual metrics: {str(e)}", exc_info=True)

    def track_error(self, trace_id: str, error: str) -> None:
        try:
            trace = self.langfuse.get_trace(trace_id)
            trace.error(
                error=error,
                metadata={
                    "component": "factual_accuracy",
                    "timestamp": datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Error tracking error event: {str(e)}", exc_info=True)

class FactualAccuracyEvaluator:
    def __init__(self, langfuse_tracker: Optional[LangfuseTracker] = None):
        try:
            self.nlp = spacy.load('en_core_web_md')
        except:
            self.nlp = spacy.load('en_core_web_sm')
        self.nlp.add_pipe('sentencizer')
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )
        self.langfuse_tracker = langfuse_tracker
        logger.info("Initialized FactualAccuracyEvaluator")

    async def _check_entailment(self, claim: str, source_text: str) -> float:
        try:
            vectors = self.vectorizer.fit_transform([claim, source_text])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error in entailment check: {str(e)}", exc_info=True)
            return 0.0

    async def evaluate_factual_accuracy(
        self, 
        research_output: ResearchToolOutput,
        trace_id: Optional[str] = None
    ) -> Tuple[float, Dict]:
        try:
            all_sources = [
                {
                    'text': content.content or content.snippet, 
                    'url': content.url
                } for content in research_output.content if content.content or content.snippet
            ]
            
            full_text = research_output.summary or " ".join(src['text'] for src in all_sources)
            
            if not full_text or not all_sources:
                return 0, {'message': 'No content to evaluate'}

            doc = self.nlp(full_text)
            claims = [sent.text.strip() for sent in doc.sents if len(sent.text.split()) > 5]
            
            if not claims:
                return 0, {'message': 'No claims to evaluate'}

            claim_scores = []
            claim_details = []
            verified_claims = 0
            unverified_claims = 0
            contradicting_claims = 0
            source_credibility_scores = []

            for claim in claims:
                max_score = 0
                best_source = None
                source_credibility = 0
                
                for source in all_sources:
                    score = await self._check_entailment(claim, source['text'])
                    if score > max_score:
                        max_score = score
                        best_source = source
                    source_credibility = max(source_credibility, score)
                
                if max_score > 0.5:
                    verified_claims += 1
                else:
                    unverified_claims += 1
                
                for other_claim in claims:
                    if other_claim != claim:
                        contradiction_score = await self._check_entailment(claim, other_claim)
                        if contradiction_score < 0.2:
                            contradicting_claims += 1
                            break
                
                claim_scores.append(max_score)
                source_credibility_scores.append(source_credibility)
                
                claim_details.append({
                    'claim': claim,
                    'score': max_score,
                    'best_source': best_source['url'] if best_source else None
                })
            
            citation_accuracy = sum(1 for score in claim_scores if score > 0.5) / len(claims) if claims else 0
            source_credibility_score = np.mean(source_credibility_scores) if source_credibility_scores else 0
            
            factual_score = (
                np.mean(claim_scores) * 0.4 +
                citation_accuracy * 0.3 +
                source_credibility_score * 0.3
            ) if claim_scores else 0
            
            fact_check_coverage = verified_claims / len(claims) if claims else 0
            
            evaluation_result = {
                'claim_details': claim_details,
                'citation_accuracy': citation_accuracy,
                'total_sources': len(all_sources),
                'contradicting_claims': contradicting_claims,
                'verified_claims': verified_claims,
                'unverified_claims': unverified_claims,
                'source_credibility_score': source_credibility_score,
                'fact_check_coverage': fact_check_coverage
            }

            if trace_id and self.langfuse_tracker:
                await self.langfuse_tracker.track_factual_metric(
                    trace_id, factual_score, evaluation_result
                )

            return factual_score, evaluation_result

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error in factual accuracy evaluation: {error_msg}", exc_info=True)
            if trace_id and self.langfuse_tracker:
                self.langfuse_tracker.track_error(trace_id, error_msg)
            return 0.0, {'error': error_msg}

def create_factual_accuracy_evaluator(
  langfuse_public_key: Optional[str] = None,
  langfuse_secret_key: Optional[str] = None,
  langfuse_host: Optional[str] = None
) -> FactualAccuracyEvaluator:
  # Use environment variables if not explicitly provided
  langfuse_public_key = langfuse_public_key or os.getenv('LANGFUSE_PUBLIC_KEY')
  langfuse_secret_key = langfuse_secret_key or os.getenv('LANGFUSE_SECRET_KEY')
  langfuse_host = langfuse_host or os.getenv('LANGFUSE_HOST')

  langfuse_tracker = None
  if langfuse_public_key and langfuse_secret_key:
      try:
          # Create configuration dictionary
          langfuse_config = {
              'public_key': langfuse_public_key,
              'secret_key': langfuse_secret_key
          }
          if langfuse_host:
              langfuse_config['host'] = langfuse_host

          # Pass configuration to LangfuseTracker
          langfuse_tracker = LangfuseTracker(
              public_key=langfuse_public_key,
              secret_key=langfuse_secret_key,
              host=langfuse_host
          )
          logger.info("Langfuse integration enabled")
      except Exception as e:
          logger.error(f"Failed to initialize Langfuse: {str(e)}")
  
  return FactualAccuracyEvaluator(langfuse_tracker=langfuse_tracker)