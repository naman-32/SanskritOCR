# Generate Training Data from Verse Images
# Segments verse images and creates labeled letter dataset
# Author: Naman Goenka
# Date: Dec 25th 2020

import sys
import os
import cv2
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segmentation.ipsegmentation.pagesegmenter import pagesegmenter


class TrainingDataGenerator:
    """
    Generates training data (letter images + ITRANS labels) from verse images.
    
    This script:
    1. Takes verse images with ITRANS ground truth
    2. Segments images into words and letters using existing segmentation logic
    3. Maps each letter image to its ITRANS class label
    4. Saves in format compatible with existing training dataset
    """
    
    def __init__(self, dict_csv_path='./classification/letter_level/dict.csv', 
                 output_dir='./training_data_augmented'):
        """
        Initialize the training data generator.
        
        Args:
            dict_csv_path: Path to dict.csv containing class index to ITRANS mapping
            output_dir: Directory to save generated training data
        """
        self.dict_csv_path = dict_csv_path
        self.output_dir = output_dir
        
        # Load class dictionary
        print(f"📚 Loading class dictionary from: {dict_csv_path}")
        self.dict_df = pd.read_csv(dict_csv_path)
        self.itrans_to_index = {row['itrans']: idx for idx, row in self.dict_df.iterrows()}
        self.num_classes = len(self.dict_df)
        print(f"   ✓ Loaded {self.num_classes} classes")
        
        # Create output directory structure
        self.setup_output_directory()
        
        # Statistics
        self.stats = {
            'images_processed': 0,
            'words_segmented': 0,
            'letters_extracted': 0,
            'letters_matched': 0,
            'letters_unmatched': 0,
            'class_distribution': {}
        }
    
    def setup_output_directory(self):
        """Create output directory structure for training data."""
        print(f"\n📁 Setting up output directory: {self.output_dir}")
        
        # Create main output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for each class
        for idx, row in self.dict_df.iterrows():
            class_name = row['itrans']
            class_dir = os.path.join(self.output_dir, f"class_{idx:03d}_{class_name}")
            os.makedirs(class_dir, exist_ok=True)
        
        # Create metadata directory
        self.metadata_dir = os.path.join(self.output_dir, 'metadata')
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        print(f"   ✓ Created {self.num_classes} class directories")
    
    def parse_itrans_word_to_letters(self, itrans_word):
        """
        Parse an ITRANS word into individual letter components.
        
        This is a simplified parser that handles common ITRANS patterns.
        For production use, you may need a more sophisticated parser.
        
        Args:
            itrans_word: Word in ITRANS format (e.g., 'dhRRitaraaShTra')
        
        Returns:
            List of letter strings
        """
        letters = []
        i = 0
        word_len = len(itrans_word)
        
        while i < word_len:
            # Check for multi-character combinations (longest first)
            matched = False
            
            # Try matching 5-character combinations
            if i + 5 <= word_len:
                substr = itrans_word[i:i+5]
                if substr in self.itrans_to_index:
                    letters.append(substr)
                    i += 5
                    matched = True
            
            # Try matching 4-character combinations
            if not matched and i + 4 <= word_len:
                substr = itrans_word[i:i+4]
                if substr in self.itrans_to_index:
                    letters.append(substr)
                    i += 4
                    matched = True
            
            # Try matching 3-character combinations
            if not matched and i + 3 <= word_len:
                substr = itrans_word[i:i+3]
                if substr in self.itrans_to_index:
                    letters.append(substr)
                    i += 3
                    matched = True
            
            # Try matching 2-character combinations
            if not matched and i + 2 <= word_len:
                substr = itrans_word[i:i+2]
                if substr in self.itrans_to_index:
                    letters.append(substr)
                    i += 2
                    matched = True
            
            # Try matching single character
            if not matched and i + 1 <= word_len:
                substr = itrans_word[i:i+1]
                if substr in self.itrans_to_index:
                    letters.append(substr)
                    i += 1
                    matched = True
            
            # If no match found, skip this character
            if not matched:
                print(f"      ⚠️  Warning: Could not match character at position {i} in '{itrans_word}'")
                i += 1
        
        return letters
    
    def process_verse_image(self, image_path, itrans_words, verse_id, two_column_layout=False):
        """
        Process a single verse image and generate training data.
        
        Args:
            image_path: Path to verse image
            itrans_words: List of ITRANS words (ground truth)
            verse_id: Unique identifier for this verse (e.g., 'gita_1_1')
            two_column_layout: Whether image has two-column layout
        
        Returns:
            Dictionary with processing statistics
        """
        print(f"\n{'='*70}")
        print(f"📸 Processing: {verse_id}")
        print(f"   Image: {image_path}")
        print(f"   Ground truth: {len(itrans_words)} words")
        print(f"{'='*70}")
        
        # Step 1: Segment image into letters
        print("\n   1. Segmenting image...")
        try:
            segmenter = pagesegmenter(image_path)
            letter_array = segmenter.get_letter_coordinates(two_column_layout=two_column_layout)
        except Exception as e:
            print(f"   ✗ Error during segmentation: {e}")
            return None
        
        num_words_segmented = len(letter_array)
        num_letters_segmented = sum(len(word) for word in letter_array)
        
        print(f"   ✓ Segmented {num_words_segmented} words, {num_letters_segmented} letters")
        
        # Step 2: Parse ITRANS words into letters
        print("\n   2. Parsing ITRANS ground truth...")
        itrans_letter_arrays = []
        total_gt_letters = 0
        
        for word_idx, itrans_word in enumerate(itrans_words):
            letters = self.parse_itrans_word_to_letters(itrans_word)
            itrans_letter_arrays.append(letters)
            total_gt_letters += len(letters)
            print(f"      Word {word_idx}: '{itrans_word}' → {letters}")
        
        print(f"   ✓ Parsed {len(itrans_words)} words into {total_gt_letters} letters")
        
        # Step 3: Match segmented letters with ITRANS labels
        print("\n   3. Matching segmented letters with ITRANS labels...")
        
        # Check if word counts match
        if num_words_segmented != len(itrans_words):
            print(f"   ⚠️  Warning: Word count mismatch!")
            print(f"      Segmented: {num_words_segmented} words")
            print(f"      Ground truth: {len(itrans_words)} words")
            print(f"      Will process min({num_words_segmented}, {len(itrans_words)}) words")
        
        # Process each word
        letters_saved = 0
        letters_skipped = 0
        
        num_words_to_process = min(num_words_segmented, len(itrans_words))
        
        for word_idx in range(num_words_to_process):
            segmented_letters = letter_array[word_idx]
            itrans_letters = itrans_letter_arrays[word_idx]
            
            # Check if letter counts match for this word
            if len(segmented_letters) != len(itrans_letters):
                print(f"      ⚠️  Word {word_idx}: Letter count mismatch (segmented: {len(segmented_letters)}, ground truth: {len(itrans_letters)})")
                print(f"         Segmented word has {len(segmented_letters)} letters")
                print(f"         Ground truth '{itrans_words[word_idx]}' has {len(itrans_letters)} letters")
                # Process min number of letters
                num_letters_to_process = min(len(segmented_letters), len(itrans_letters))
            else:
                num_letters_to_process = len(segmented_letters)
            
            # Save each letter with its label
            for letter_idx in range(num_letters_to_process):
                itrans_label = itrans_letters[letter_idx]
                
                # Get class index
                if itrans_label not in self.itrans_to_index:
                    print(f"      ⚠️  Warning: ITRANS label '{itrans_label}' not in dictionary, skipping")
                    letters_skipped += 1
                    continue
                
                class_idx = self.itrans_to_index[itrans_label]
                
                # Load letter image
                letter_img_path = f"./tmp_letters/{word_idx}-{letter_idx}.png"
                
                if not os.path.exists(letter_img_path):
                    print(f"      ⚠️  Warning: Letter image not found: {letter_img_path}")
                    letters_skipped += 1
                    continue
                
                letter_img = cv2.imread(letter_img_path)
                if letter_img is None:
                    print(f"      ⚠️  Warning: Could not read letter image: {letter_img_path}")
                    letters_skipped += 1
                    continue
                
                # Save to appropriate class directory
                class_dir = os.path.join(self.output_dir, f"class_{class_idx:03d}_{itrans_label}")
                output_filename = f"{verse_id}_w{word_idx}_l{letter_idx}.png"
                output_path = os.path.join(class_dir, output_filename)
                
                cv2.imwrite(output_path, letter_img)
                letters_saved += 1
                
                # Update statistics
                if itrans_label not in self.stats['class_distribution']:
                    self.stats['class_distribution'][itrans_label] = 0
                self.stats['class_distribution'][itrans_label] += 1
        
        print(f"\n   ✓ Saved {letters_saved} letter images")
        if letters_skipped > 0:
            print(f"   ⚠️  Skipped {letters_skipped} letters due to errors")
        
        # Update global statistics
        self.stats['images_processed'] += 1
        self.stats['words_segmented'] += num_words_segmented
        self.stats['letters_extracted'] += num_letters_segmented
        self.stats['letters_matched'] += letters_saved
        self.stats['letters_unmatched'] += letters_skipped
        
        # Save metadata for this verse
        metadata = {
            'verse_id': verse_id,
            'image_path': image_path,
            'words_segmented': num_words_segmented,
            'letters_segmented': num_letters_segmented,
            'letters_saved': letters_saved,
            'letters_skipped': letters_skipped,
            'ground_truth_words': itrans_words
        }
        
        return metadata
    
    def process_ground_truth_file(self, ground_truth_file, images_dir, two_column_layout=False):
        """
        Process all verses from a ground truth file.
        
        Args:
            ground_truth_file: Path to ground truth file (format: image_name,word1,word2,...)
            images_dir: Directory containing verse images
            two_column_layout: Whether images have two-column layout
        """
        print(f"\n{'='*70}")
        print(f"🚀 TRAINING DATA GENERATION")
        print(f"{'='*70}")
        print(f"\n📂 Ground truth file: {ground_truth_file}")
        print(f"📂 Images directory: {images_dir}")
        print(f"📂 Output directory: {self.output_dir}")
        
        # Load ground truth
        ground_truth = {}
        try:
            with open(ground_truth_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split(',')
                    if len(parts) >= 2:
                        image_name = parts[0]
                        words = parts[1:]
                        ground_truth[image_name] = words
        except FileNotFoundError:
            print(f"\n✗ Error: Ground truth file not found: {ground_truth_file}")
            return
        
        print(f"\n✓ Loaded ground truth for {len(ground_truth)} verses")
        
        # Process each verse
        all_metadata = []
        
        for image_name, itrans_words in ground_truth.items():
            image_path = os.path.join(images_dir, image_name)
            
            if not os.path.exists(image_path):
                print(f"\n⚠️  Warning: Image not found: {image_path}, skipping")
                continue
            
            # Generate verse ID from image name
            verse_id = os.path.splitext(image_name)[0]
            
            # Process this verse
            metadata = self.process_verse_image(
                image_path=image_path,
                itrans_words=itrans_words,
                verse_id=verse_id,
                two_column_layout=two_column_layout
            )
            
            if metadata:
                all_metadata.append(metadata)
        
        # Save all metadata
        self.save_metadata(all_metadata)
        
        # Print final statistics
        self.print_statistics()
    
    def save_metadata(self, all_metadata):
        """Save metadata for all processed verses."""
        metadata_file = os.path.join(self.metadata_dir, 'generation_metadata.txt')
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("TRAINING DATA GENERATION METADATA\n")
            f.write("="*70 + "\n\n")
            
            for metadata in all_metadata:
                f.write(f"Verse ID: {metadata['verse_id']}\n")
                f.write(f"  Image: {metadata['image_path']}\n")
                f.write(f"  Words segmented: {metadata['words_segmented']}\n")
                f.write(f"  Letters segmented: {metadata['letters_segmented']}\n")
                f.write(f"  Letters saved: {metadata['letters_saved']}\n")
                f.write(f"  Letters skipped: {metadata['letters_skipped']}\n")
                f.write(f"  Ground truth: {', '.join(metadata['ground_truth_words'])}\n")
                f.write("\n")
        
        print(f"\n💾 Metadata saved to: {metadata_file}")
    
    def print_statistics(self):
        """Print final statistics."""
        print(f"\n{'='*70}")
        print(f"📊 GENERATION STATISTICS")
        print(f"{'='*70}")
        
        print(f"\n📈 Overall:")
        print(f"   Images processed: {self.stats['images_processed']}")
        print(f"   Words segmented: {self.stats['words_segmented']}")
        print(f"   Letters extracted: {self.stats['letters_extracted']}")
        print(f"   Letters matched: {self.stats['letters_matched']}")
        print(f"   Letters unmatched: {self.stats['letters_unmatched']}")
        
        if self.stats['letters_extracted'] > 0:
            match_rate = (self.stats['letters_matched'] / self.stats['letters_extracted']) * 100
            print(f"   Match rate: {match_rate:.2f}%")
        
        print(f"\n📊 Class Distribution (Top 20):")
        sorted_classes = sorted(self.stats['class_distribution'].items(), 
                               key=lambda x: x[1], reverse=True)
        
        for i, (class_name, count) in enumerate(sorted_classes[:20]):
            print(f"   {i+1:2d}. {class_name:15s}: {count:4d} samples")
        
        if len(sorted_classes) > 20:
            print(f"   ... and {len(sorted_classes) - 20} more classes")
        
        print(f"\n✅ Training data generation complete!")
        print(f"📁 Output directory: {self.output_dir}")


def main():
    """Main function to run training data generation."""
    
    # Configuration
    DICT_CSV_PATH = './classification/letter_level/dict.csv'
    GROUND_TRUTH_FILE = './verses/ground_truth.txt'
    IMAGES_DIR = './verses/'
    OUTPUT_DIR = './training_data_augmented'
    TWO_COLUMN_LAYOUT = False  # Set to True if your images have two-column layout
    
    # Create generator
    generator = TrainingDataGenerator(
        dict_csv_path=DICT_CSV_PATH,
        output_dir=OUTPUT_DIR
    )
    
    # Process all verses
    generator.process_ground_truth_file(
        ground_truth_file=GROUND_TRUTH_FILE,
        images_dir=IMAGES_DIR,
        two_column_layout=TWO_COLUMN_LAYOUT
    )
    
    print("\n" + "="*70)
    print("🎉 DONE!")
    print("="*70)
    print(f"\nYour augmented training data is ready in: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Review the generated data in class directories")
    print("2. Merge with existing training dataset")
    print("3. Retrain your CNN model with augmented data")
    print("4. Evaluate improved model performance")


if __name__ == "__main__":
    main()
