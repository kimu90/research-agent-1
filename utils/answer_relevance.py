import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tools.research.common.model_schemas import ResearchToolOutput

nlp = spacy.blank('en')
nlp.add_pipe('sentencizer')

class AnswerRelevanceEvaluator:
    def __init__(self):
        self.nlp = nlp
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

    def calculate_similarity(self, text1, text2):
        vectors = self.vectorizer.fit_transform([text1, text2])
        return cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    def evaluate_answer_relevance(self, research_output: ResearchToolOutput, query: str):
        full_text = research_output.summary or " ".join(content.content for content in research_output.content)
        similarity = self.calculate_similarity(query, full_text)
        
        query_doc = self.nlp(query)
        answer_doc = self.nlp(full_text)
        
        query_keywords = set(token.lemma_.lower() for token in query_doc if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop)
        answer_keywords = set(token.lemma_.lower() for token in answer_doc if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop)
        
        keyword_overlap = len(query_keywords.intersection(answer_keywords)) / len(query_keywords) if query_keywords else 0
        
        score = (similarity + keyword_overlap) / 2
        
        return score, {
            'similarity': similarity,
            'keyword_overlap': keyword_overlap
        }

def create_answer_relevance_evaluator():
    return AnswerRelevanceEvaluator()