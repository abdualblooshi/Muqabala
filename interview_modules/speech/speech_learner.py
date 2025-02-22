import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Tuple
import pickle

class SpeechLearner:
    def __init__(self, model_dir: str = "speech_models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        
        # Load existing corrections if any
        self.corrections_file = os.path.join(model_dir, "corrections.json")
        self.corrections = self._load_corrections()
        
        # Load vectorizer model if exists
        self.vectorizer_file = os.path.join(model_dir, "vectorizer.pkl")
        if os.path.exists(self.vectorizer_file):
            with open(self.vectorizer_file, 'rb') as f:
                self.vectorizer = pickle.load(f)
        
        self.logger = logging.getLogger(__name__)
    
    def _load_corrections(self) -> Dict[str, List[Dict]]:
        """Load existing corrections from file."""
        if os.path.exists(self.corrections_file):
            try:
                with open(self.corrections_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading corrections: {e}")
                return {}
        return {}
    
    def _save_corrections(self):
        """Save corrections to file."""
        try:
            with open(self.corrections_file, 'w') as f:
                json.dump(self.corrections, f, indent=2)
            
            # Save vectorizer model
            with open(self.vectorizer_file, 'wb') as f:
                pickle.dump(self.vectorizer, f)
        except Exception as e:
            self.logger.error(f"Error saving corrections: {e}")
    
    def add_correction(self, original: str, corrected: str, context: str = ""):
        """Add a new correction pair to the dataset."""
        if not original or not corrected:
            return
        
        correction_entry = {
            "original": original,
            "corrected": corrected,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store by original text as key
        if original not in self.corrections:
            self.corrections[original] = []
        self.corrections[original].append(correction_entry)
        
        # Update vectorizer with new data
        all_texts = [original, corrected]
        self.vectorizer.fit(all_texts)
        
        self._save_corrections()
    
    def get_similar_corrections(self, text: str, threshold: float = 0.7) -> List[Dict]:
        """Find similar corrections from the dataset."""
        if not text or not self.corrections:
            return []
        
        try:
            # Transform the input text and all stored corrections
            text_vector = self.vectorizer.transform([text])
            similar_corrections = []
            
            for original, corrections in self.corrections.items():
                original_vector = self.vectorizer.transform([original])
                similarity = cosine_similarity(text_vector, original_vector)[0][0]
                
                if similarity >= threshold:
                    for correction in corrections:
                        similar_corrections.append({
                            "similarity": similarity,
                            "correction": correction
                        })
            
            # Sort by similarity
            similar_corrections.sort(key=lambda x: x["similarity"], reverse=True)
            return similar_corrections
            
        except Exception as e:
            self.logger.error(f"Error finding similar corrections: {e}")
            return []
    
    def suggest_correction(self, text: str, context: str = "") -> Tuple[str, float]:
        """Suggest a correction based on learning history."""
        similar = self.get_similar_corrections(text)
        
        if not similar:
            return text, 0.0
        
        # Get the most similar correction
        best_match = similar[0]
        if best_match["similarity"] > 0.8:  # High confidence threshold
            return best_match["correction"]["corrected"], best_match["similarity"]
        
        return text, best_match["similarity"]
    
    def get_correction_stats(self) -> Dict:
        """Get statistics about the corrections dataset."""
        return {
            "total_corrections": sum(len(corr) for corr in self.corrections.values()),
            "unique_originals": len(self.corrections),
            "last_updated": max(
                (c["timestamp"] for corr in self.corrections.values() for c in corr),
                default=None
            )
        } 