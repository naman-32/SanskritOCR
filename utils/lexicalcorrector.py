## Lexical Corrector for Sanskrit OCR using Levenshtein Distance implemented using DP
## Author: Naman Goenka
## Date: Dec 25th 2020

import logging
import time
from typing import List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LexicalCorrector:
    """
    Post-OCR lexical correction using dictionary lookup and Levenshtein distance.
    Corrects OCR errors by finding the closest matching word in a Sanskrit dictionary.
    """
    
    def __init__(self, dictionary_path: str, max_distance: int = 2):
        """
        Initialize the lexical corrector.
        
        Args:
            dictionary_path: Path to the Sanskrit dictionary file (one word per line in ITRANS)
            max_distance: Maximum Levenshtein distance for correction (default: 2)
        """
        self.dictionary_path = dictionary_path
        self.max_distance = max_distance
        self.dictionary = set()
        self.dictionary_list = []
        
        # Load dictionary
        self._load_dictionary()
        
        logger.info(f"Successfully loaded dictionary with {len(self.dictionary)} entries")
    
    def _load_dictionary(self):
        """Load Sanskrit dictionary from file."""
        try:
            with open(self.dictionary_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith('#'):
                        self.dictionary.add(word)
                        self.dictionary_list.append(word)
            
            logger.info(f"Loaded {len(self.dictionary)} words from dictionary")
        except FileNotFoundError:
            logger.error(f"Dictionary file not found: {self.dictionary_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading dictionary: {e}")
            raise
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Levenshtein distance (number of edits needed)
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def find_closest_match(self, word: str) -> Tuple[Optional[str], int]:
        """
        Find the closest matching word in the dictionary.
        
        Args:
            word: Input word to correct
            
        Returns:
            Tuple of (closest_match, distance) or (None, -1) if no match within max_distance
        """
        # If word is already in dictionary, return it
        if word in self.dictionary:
            return word, 0
        
        min_distance = float('inf')
        closest_match = None
        
        # Search through dictionary for closest match
        for dict_word in self.dictionary_list:
            distance = self.levenshtein_distance(word, dict_word)
            
            if distance < min_distance:
                min_distance = distance
                closest_match = dict_word
                
                # Early exit if perfect match found
                if distance == 0:
                    break
        
        # Only return match if within max_distance threshold
        if min_distance <= self.max_distance:
            return closest_match, min_distance
        else:
            return None, -1
    
    def correct_word(self, word: str, verbose: bool = False) -> Tuple[str, bool, int]:
        """
        Correct a single word using dictionary lookup.
        
        Args:
            word: Word to correct
            verbose: If True, log correction details
            
        Returns:
            Tuple of (corrected_word, was_corrected, distance)
        """
        if not word or len(word) == 0:
            return word, False, -1
        
        # Find closest match
        closest_match, distance = self.find_closest_match(word)
        
        if closest_match and closest_match != word:
            if verbose:
                logger.info(f"Corrected '{word}' -> '{closest_match}' (distance: {distance})")
            return closest_match, True, distance
        else:
            return word, False, 0
    
    def correct_words(self, words: List[str], verbose: bool = True) -> Tuple[List[str], List[dict]]:
        """
        Correct a list of words using dictionary lookup.
        
        Args:
            words: List of words to correct
            verbose: If True, log correction details
            
        Returns:
            Tuple of (corrected_words, corrections_info)
            where corrections_info is a list of dicts with correction details
        """
        corrected_words = []
        corrections_info = []
        
        start_time = time.time()
        
        for i, word in enumerate(words):
            corrected_word, was_corrected, distance = self.correct_word(word, verbose=False)
            corrected_words.append(corrected_word)
            
            if was_corrected:
                correction_info = {
                    'index': i,
                    'original': word,
                    'corrected': corrected_word,
                    'distance': distance
                }
                corrections_info.append(correction_info)
                
                if verbose:
                    logger.info(f"Corrected '{word}' -> '{corrected_word}' (distance: {distance})")
        
        elapsed_time = time.time() - start_time
        
        if verbose and corrections_info:
            logger.info(f"Applied {len(corrections_info)} corrections in {elapsed_time:.3f}s")
            logger.info(f"Processing speed: {len(words)/elapsed_time:.1f} words/second")
        
        return corrected_words, corrections_info
    
    def get_correction_stats(self, corrections_info: List[dict]) -> dict:
        """
        Get statistics about corrections applied.
        
        Args:
            corrections_info: List of correction info dicts
            
        Returns:
            Dictionary with correction statistics
        """
        if not corrections_info:
            return {
                'total_corrections': 0,
                'avg_distance': 0,
                'max_distance': 0,
                'min_distance': 0
            }
        
        distances = [c['distance'] for c in corrections_info]
        
        return {
            'total_corrections': len(corrections_info),
            'avg_distance': sum(distances) / len(distances),
            'max_distance': max(distances),
            'min_distance': min(distances)
        }


# Example usage
if __name__ == "__main__":
    # Initialize corrector
    corrector = LexicalCorrector(
        dictionary_path='./dictionary/sanskrit_dictionary.txt',
        max_distance=2
    )
    
    # Test words
    test_words = ['dhRRitaraaShTra', 'uvaca', 'dharmakShetre', 'kurukShetre']
    
    print("\nTesting Lexical Correction:")
    print("="*70)
    
    for word in test_words:
        corrected, was_corrected, distance = corrector.correct_word(word, verbose=True)
        print(f"  {word:20} -> {corrected:20} (corrected: {was_corrected}, distance: {distance})")
    
    print("\n" + "="*70)
