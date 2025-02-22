from transformers import pipeline
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EvaluationMetrics:
    relevance_score: float
    confidence_score: float
    technical_score: float
    language_score: float
    overall_score: float
    feedback: str

class InterviewEvaluator:
    def __init__(self):
        # Initialize sentiment analysis pipeline
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        
        # Initialize zero-shot classification pipeline
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
    def evaluate_response(self, 
                         question: str, 
                         response: str,
                         expected_keywords: List[str] = None) -> EvaluationMetrics:
        """
        Evaluate an interview response based on multiple criteria.
        """
        # Analyze sentiment and confidence
        sentiment_result = self.sentiment_analyzer(response)[0]
        confidence_score = float(sentiment_result['score'])
        
        # Evaluate technical relevance
        if expected_keywords:
            relevance_score = self._calculate_keyword_relevance(response, expected_keywords)
        else:
            relevance_score = self._evaluate_relevance(question, response)
        
        # Evaluate language proficiency
        language_score = self._evaluate_language_quality(response)
        
        # Calculate technical score
        technical_score = self._evaluate_technical_depth(response)
        
        # Calculate overall score (weighted average)
        overall_score = np.mean([
            relevance_score * 0.3,
            confidence_score * 0.2,
            technical_score * 0.3,
            language_score * 0.2
        ])
        
        # Generate feedback
        feedback = self._generate_feedback(
            relevance_score,
            confidence_score,
            technical_score,
            language_score
        )
        
        return EvaluationMetrics(
            relevance_score=relevance_score,
            confidence_score=confidence_score,
            technical_score=technical_score,
            language_score=language_score,
            overall_score=overall_score,
            feedback=feedback
        )
    
    def _calculate_keyword_relevance(self, 
                                   response: str, 
                                   keywords: List[str]) -> float:
        """Calculate relevance based on keyword presence."""
        response_lower = response.lower()
        matched_keywords = sum(1 for keyword in keywords 
                             if keyword.lower() in response_lower)
        return matched_keywords / len(keywords) if keywords else 0.0
    
    def _evaluate_relevance(self, 
                           question: str, 
                           response: str) -> float:
        """Evaluate response relevance using zero-shot classification."""
        hypothesis = f"This is a relevant answer to: {question}"
        result = self.classifier(
            response,
            candidate_labels=["relevant", "irrelevant"]
        )
        return result['scores'][result['labels'].index("relevant")]
    
    def _evaluate_language_quality(self, text: str) -> float:
        """Evaluate language proficiency."""
        # Simple metrics for demonstration
        words = text.split()
        avg_word_length = np.mean([len(word) for word in words])
        vocabulary_size = len(set(words))
        
        # Normalize scores
        length_score = min(avg_word_length / 10, 1.0)
        vocab_score = min(vocabulary_size / 100, 1.0)
        
        return np.mean([length_score, vocab_score])
    
    def _evaluate_technical_depth(self, response: str) -> float:
        """Evaluate technical depth of the response."""
        technical_indicators = [
            "algorithm", "implementation", "system", "architecture",
            "database", "framework", "optimization", "complexity",
            "design pattern", "scalability"
        ]
        
        # Calculate technical term density
        words = response.lower().split()
        technical_term_count = sum(1 for indicator in technical_indicators 
                                 if indicator in response.lower())
        
        return min(technical_term_count / 5, 1.0)  # Normalize to [0,1]
    
    def _generate_feedback(self,
                          relevance: float,
                          confidence: float,
                          technical: float,
                          language: float) -> str:
        """Generate human-readable feedback."""
        feedback_parts = []
        
        if relevance < 0.5:
            feedback_parts.append("Try to focus more on directly answering the question.")
        elif relevance >= 0.8:
            feedback_parts.append("Excellent relevance to the question.")
            
        if confidence < 0.5:
            feedback_parts.append("Consider speaking with more confidence.")
        elif confidence >= 0.8:
            feedback_parts.append("Good confidence level in responses.")
            
        if technical < 0.5:
            feedback_parts.append("Include more technical details in your answers.")
        elif technical >= 0.8:
            feedback_parts.append("Strong technical depth demonstrated.")
            
        if language < 0.5:
            feedback_parts.append("Work on improving clarity and articulation.")
        elif language >= 0.8:
            feedback_parts.append("Excellent communication skills.")
            
        return " ".join(feedback_parts) 