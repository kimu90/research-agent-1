import spacy
import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.research.common.model_schemas import ResearchToolOutput

nlp = spacy.blank('en')
nlp.add_pipe('sentencizer')

class FactualAccuracyEvaluator:
    def __init__(self):
        self.nlp = nlp
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )

    def _check_entailment(self, claim: str, source_text: str) -> float:
        claim_doc = self.nlp(claim)
        source_doc = self.nlp(source_text)
        similarity = claim_doc.similarity(source_doc)
        return min(max(similarity, 0), 1)

    def evaluate_factual_accuracy(self, research_output: ResearchToolOutput) -> Tuple[float, Dict]:
        all_sources = [
            {
                'text': content.content or content.snippet, 
                'url': content.url
            } for content in research_output.content
        ]
        
        full_text = research_output.summary or " ".join(
            content.content for content in research_output.content
        )
        
        doc = self.nlp(full_text)
        claims = [sent.text for sent in doc.sents if any(token.dep_ == 'ROOT' for token in sent)]
        
        claim_scores = []
        claim_details = []
        verified_claims = 0
        unverified_claims = 0
        contradicting_claims = 0
        source_credibility_scores = []
        
        if not claims:
            return 0, {'message': 'No claims to evaluate'}

        for idx, claim in enumerate(claims, 1):
            max_score = 0
            best_source = None
            source_credibility = 0
            
            if not all_sources:  # Check if there are no sources
                continue

            for source in all_sources:
                score = self._check_entailment(claim, source['text'])
                if score > max_score:
                    max_score = score
                    best_source = source
                source_credibility = max(source_credibility, score)
            
            if max_score > 0.7:
                verified_claims += 1
            else:
                unverified_claims += 1
            
            # Check for contradictions between claims
            for other_claim in claims:
                if other_claim != claim:
                    contradiction_score = self._check_entailment(claim, other_claim)
                    if contradiction_score < 0.3:
                        contradicting_claims += 1
                        break
            
            claim_scores.append(max_score)
            source_credibility_scores.append(source_credibility)
            
            claim_details.append({
                'claim': claim,
                'score': max_score,
                'best_source': best_source['url'] if best_source else None
            })
        
        # Check if claim_scores are empty
        citation_accuracy = sum(1 for score in claim_scores if score > 0.7) / len(claims) if claims else 0
        factual_score = np.mean(claim_scores) * 0.7 + citation_accuracy * 0.3 if claim_scores else 0
        source_credibility_score = np.mean(source_credibility_scores) if source_credibility_scores else 0
        fact_check_coverage = (verified_claims / len(claims)) if claims else 0
        
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
