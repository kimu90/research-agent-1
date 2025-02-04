import spacy
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.research.common.model_schemas import ResearchToolOutput

try:
    nlp = spacy.load('en_core_web_md')
except:
    nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('sentencizer')

class FactualAccuracyEvaluator:
    def __init__(self):
        self.nlp = nlp
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )

    def _check_entailment(self, claim: str, source_text: str) -> float:
        texts = [claim, source_text]
        try:
            vectors = self.vectorizer.fit_transform(texts)
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error in entailment check: {e}")
            return 0.0

    def evaluate_factual_accuracy(self, research_output: ResearchToolOutput) -> Tuple[float, Dict]:
        all_sources = [
            {
                'text': content.content or content.snippet, 
                'url': content.url
            } for content in research_output.content if content.content or content.snippet
        ]
        
        full_text = research_output.summary or " ".join(
            src['text'] for src in all_sources
        )
        
        if not full_text or not all_sources:
            return 0, {'message': 'No content to evaluate'}

        doc = self.nlp(full_text)
        claims = [sent.text.strip() for sent in doc.sents 
                 if len(sent.text.split()) > 5]  # Only consider substantial sentences
        
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
                score = self._check_entailment(claim, source['text'])
                if score > max_score:
                    max_score = score
                    best_source = source
                source_credibility = max(source_credibility, score)
            
            if max_score > 0.5:  # Lowered threshold from 0.7
                verified_claims += 1
            else:
                unverified_claims += 1
            
            for other_claim in claims:
                if other_claim != claim:
                    contradiction_score = self._check_entailment(claim, other_claim)
                    if contradiction_score < 0.2:  # Adjusted from 0.3
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
        
        # Updated scoring weights
        factual_score = (
            np.mean(claim_scores) * 0.4 +  # Reduced from 0.7
            citation_accuracy * 0.3 +       # Same weight
            source_credibility_score * 0.3  # Added credibility weight
        ) if claim_scores else 0
        
        fact_check_coverage = verified_claims / len(claims) if claims else 0
        
        return factual_score, {
            'claim_details': claim_details,
            'citation_accuracy': citation_accuracy,
            'total_sources': len(all_sources),
            'contradicting_claims': contradicting_claims,
            'verified_claims': verified_claims,
            'unverified_claims': unverified_claims,
            'source_credibility_score': source_credibility_score,
            'fact_check_coverage': fact_check_coverage
        }

def create_factual_accuracy_evaluator() -> FactualAccuracyEvaluator:
    return FactualAccuracyEvaluator()