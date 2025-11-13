# Labeled Letter Dataset Generator from Verse Images
# Segments images and maps letters to ITRANS class labels
# Author: Naman Goenka
# Date: Dec 25th 2020

import sys
import os
import cv2
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from segmentation.ipsegmentation.pagesegmenter import pagesegmenter


class DatasetGenerator:
    """Generate training dataset from verse images."""
    
    def __init__(self, dict_csv='../classification/letter_level/dict.csv'):
        """Initialize with class dictionary."""
        self.dict_df = pd.read_csv(dict_csv)
        self.itrans_to_index = {row['itrans']: idx 
                                for idx, row in self.dict_df.iterrows()}
        self.num_classes = len(self.dict_df)
        
    def parse_itrans_word(self, word):
        """Parse ITRANS word into letter components."""
        letters = []
        i = 0
        
        while i < len(word):
            matched = False
            
            # Try matching 5, 4, 3, 2, 1 character combinations
            for length in [5, 4, 3, 2, 1]:
                if i + length <= len(word):
                    substr = word[i:i+length]
                    if substr in self.itrans_to_index:
                        letters.append(substr)
                        i += length
                        matched = True
                        break
            
            if not matched:
                i += 1
        
        return letters
    
    def process_verse(self, image_path, itrans_words, verse_id, output_dir):
        """Process single verse and generate letter images."""
        # Segment image
        segmenter = pagesegmenter(image_path)
        letter_array = segmenter.get_letter_coordinates(two_column_layout=False)
        
        num_words = len(letter_array)
        num_letters = sum(len(word) for word in letter_array)
        
        # Parse ITRANS words
        itrans_letters = [self.parse_itrans_word(w) for w in itrans_words]
        
        # Match and save letters
        saved = 0
        num_to_process = min(num_words, len(itrans_words))
        
        for word_idx in range(num_to_process):
            seg_letters = letter_array[word_idx]
            gt_letters = itrans_letters[word_idx]
            
            num_letters_to_save = min(len(seg_letters), len(gt_letters))
            
            for letter_idx in range(num_letters_to_save):
                label = gt_letters[letter_idx]
                
                if label not in self.itrans_to_index:
                    continue
                
                class_idx = self.itrans_to_index[label]
                
                # Load letter image
                letter_path = f"./tmp_letters/{word_idx}-{letter_idx}.png"
                if not os.path.exists(letter_path):
                    continue
                
                letter_img = cv2.imread(letter_path)
                if letter_img is None:
                    continue
                
                # Save to class directory
                class_dir = os.path.join(output_dir, f"class_{class_idx:03d}_{label}")
                os.makedirs(class_dir, exist_ok=True)
                
                output_file = f"{verse_id}_w{word_idx}_l{letter_idx}.png"
                cv2.imwrite(os.path.join(class_dir, output_file), letter_img)
                saved += 1
        
        return saved
    
    def generate(self, ground_truth_file, images_dir, output_dir, max_verses=None):
        """Generate dataset from ground truth."""
        # Load ground truth
        ground_truth = {}
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    ground_truth[parts[0]] = parts[1:]
        
        if max_verses:
            ground_truth = dict(list(ground_truth.items())[:max_verses])
        
        print(f"Processing {len(ground_truth)} verses")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each verse
        total_saved = 0
        for img_name, words in ground_truth.items():
            img_path = os.path.join(images_dir, img_name)
            if not os.path.exists(img_path):
                continue
            
            verse_id = os.path.splitext(img_name)[0]
            saved = self.process_verse(img_path, words, verse_id, output_dir)
            total_saved += saved
            
            print(f"  {img_name}: {saved} letters")
        
        print(f"\nTotal letters saved: {total_saved}")
        return total_saved


def main():
    """Main dataset generation workflow."""
    print("Dataset Generation from Gita Verses")
    print("-" * 50)
    
    # Clean up existing dataset and temp directories
    import shutil
    for dir_path in ['sample_dataset', '../tmp_words', '../tmp_letters']:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Cleaned up {dir_path}")
    
    generator = DatasetGenerator()
    
    # Generate from first 10 verses
    generator.generate(
        ground_truth_file='ground_truth.txt',
        images_dir='sample_images',
        output_dir='sample_dataset',
        max_verses=10
    )
    
    print("\nDone. Dataset saved to sample_dataset/")


if __name__ == "__main__":
    main()
