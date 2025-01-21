# utils/evaluation.py
import pdb
import spacy
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from tools.research.common.model_schemas import ResearchToolOutput, ContentItem

class FactualAccuracyEvaluator:
    def __init__(self, nlp_model: str = 'en_core_web_sm'):
        """
        Initialize the Factual Accuracy Evaluator
        
        Args:
            nlp_model (str): spaCy language model to use for NLP tasks
        """
          # Breakpoint at initialization
        try:
            logging.info(f"Attempting to load spaCy model: {nlp_model}")
            self.nlp = spacy.load(nlp_model)
            logging.info(f"Successfully loaded spaCy model: {nlp_model}")
        except Exception as e:
            logging.error(f"Failed to load spaCy model: {e}")
              # Breakpoint if model loading fails
            raise

    def _check_entailment(self, claim: str, source_text: str) -> float:
        """
        Check entailment between a claim and source text
        
        Args:
            claim (str): The claim to verify
            source_text (str): The source text to check against
        
        Returns:
            float: Entailment score between 0 and 1
        """
          # Breakpoint at entailment check start
        try:
            logging.info(f"Performing entailment check for claim: {claim[:50]}...")
            
            claim_doc = self.nlp(claim)
            source_doc = self.nlp(source_text)
            
            # Basic similarity calculation
            similarity = claim_doc.similarity(source_doc)
            
            # Add logging for similarity score
            logging.info(f"Entailment similarity score: {similarity}")
            
              # Breakpoint after similarity calculation
            
            return min(max(similarity, 0), 1)
        except Exception as e:
            logging.warning(f"Entailment check failed: {e}")
              # Breakpoint if entailment check fails
            return 0.0

    def evaluate_factual_accuracy(self, research_output: ResearchToolOutput) -> Tuple[float, Dict]:
        """
        Evaluate factual accuracy of research output
        
        Args:
            research_output (ResearchToolOutput): Output from research tool
        
        Returns:
            Tuple[float, Dict]: Factual accuracy score and detailed analysis
        """
          # Initial breakpoint for factual accuracy evaluation
        
        # Logging start of evaluation
        logging.info("Starting factual accuracy evaluation")
        
        # Prepare sources from content
        all_sources = [
            {
                'text': content.content or content.snippet, 
                'url': content.url
            } for content in research_output.content
        ]
        
        logging.info(f"Total sources prepared: {len(all_sources)}")
          # Breakpoint after source preparation
        
        # Concatenate text for claim extraction
        full_text = research_output.summary or " ".join(
            content.content for content in research_output.content
        )
        
        # Extract claims using spaCy
        doc = self.nlp(full_text)
        claims = [
            sent.text for sent in doc.sents 
            if any(token.dep_ == 'ROOT' for token in sent)
        ]
        
        logging.info(f"Total claims extracted: {len(claims)}")
          # Breakpoint after claim extraction
        
        claim_scores = []
        claim_details = []
        
        for idx, claim in enumerate(claims, 1):
            logging.info(f"Processing claim {idx}/{len(claims)}: {claim[:50]}...")
            
            # Find best matching source
            max_score = 0
            best_source = None
            
            for source in all_sources:
                score = self._check_entailment(claim, source['text'])
                if score > max_score:
                    max_score = score
                    best_source = source
            
            claim_scores.append(max_score)
            claim_details.append({
                'claim': claim,
                'score': max_score,
                'best_source': best_source['url'] if best_source else None
            })
            
              # Breakpoint for each claim processing
        
        # Calculate accuracy metrics
        citation_accuracy = sum(1 for score in claim_scores if score > 0.7) / len(claims) if claims else 0
        factual_score = np.mean(claim_scores) * 0.7 + citation_accuracy * 0.3
        
        logging.info(f"Factual Score: {factual_score}")
        logging.info(f"Citation Accuracy: {citation_accuracy}")
        
          # Final breakpoint before returning results
        
        return factual_score, {
            'claim_details': claim_details,
            'citation_accuracy': citation_accuracy,
            'total_sources': len(all_sources)
        }

def create_factual_accuracy_evaluator() -> FactualAccuracyEvaluator:
    """
    Factory function to create a FactualAccuracyEvaluator
    
    Returns:
        FactualAccuracyEvaluator: An instance of the evaluator
    """
      # Breakpoint in factory function
    return FactualAccuracyEvaluator()