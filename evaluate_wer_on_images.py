# WER Evaluation Script for Sanskrit OCR
# Evaluates OCR accuracy with and without lexical correction
# verses/ground_truth.txt is used for ground truth by default
# Format: image_filename (in verse folder) ,word1,word2,word3,...
# Author: Naman Goenka
# Date: Dec 25th 2020

import sys
import os
import time

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from segmentation.ipsegmentation.pagesegmenter import pagesegmenter
from classification.letter_level.classifier import OCRclassifier
from utils.textconverter import textconverter
from utils.image_preprocessor import ImagePreprocessor


def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate (WER) using Levenshtein distance at word level.
    
    Args:
        reference: List of reference words (ground truth)
        hypothesis: List of hypothesis words (OCR output)
    
    Returns:
        WER as percentage, number of errors, accuracy
    """
    # Levenshtein distance at word level
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    
    if ref_len == 0:
        return 0.0 if hyp_len == 0 else 100.0, 0, 100.0
    
    # Create distance matrix
    d = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
    
    # Initialize first row and column
    for i in range(ref_len + 1):
        d[i][0] = i
    for j in range(hyp_len + 1):
        d[0][j] = j
    
    # Calculate distances
    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i-1] == hypothesis[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion = d[i][j-1] + 1
                deletion = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)
    
    errors = d[ref_len][hyp_len]
    wer = (errors / ref_len) * 100
    accuracy = max(0, 100 - wer)
    
    return wer, errors, accuracy


def load_ground_truth(ground_truth_file):
    """
    Load ground truth from file.
    Format: image_filename,word1,word2,word3,...
    """
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
        
        return ground_truth
    except FileNotFoundError:
        print(f"Error: Ground truth file not found: {ground_truth_file}")
        return {}


def evaluate_on_test_images(test_images_dir='./verses/', ground_truth_file='./verses/ground_truth.txt'):
    """
    Run complete OCR pipeline on test images and evaluate WER.
    """
    print("\n" + "="*70)
    print("SANSKRIT OCR - WER EVALUATION ON TEST IMAGES")
    print("="*70)
    print("\nThis script runs the complete OCR pipeline (segmentation + classification)")
    print("on test images and compares output with and without lexical correction.")
    
    # Clean up temporary directories from previous runs
    import shutil
    for tmp_dir in ['./tmp_words', './tmp_letters']:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
    
    # Load ground truth
    print("\n" + "="*70)
    print("WER EVALUATION: Full OCR Pipeline on Test Images")
    print("="*70)
    
    print(f"\nLoading ground truth from: {ground_truth_file}")
    ground_truth = load_ground_truth(ground_truth_file)
    print(f"Loaded ground truth for {len(ground_truth)} images")
    
    # Find test images
    test_images = []
    for img_name in ground_truth.keys():
        img_path = os.path.join(test_images_dir, img_name)
        if os.path.exists(img_path):
            test_images.append((img_name, img_path))
    
    print(f"\nFound {len(test_images)} test images with ground truth")
    
    if not test_images:
        print("\nNo test images found!")
        return
    
    # Initialize OCR components
    print("\nInitializing OCR components...")
    
    # Image preprocessor for handling degraded manuscripts
    preprocessor = ImagePreprocessor(enable_super_resolution=False)
    print("Image preprocessor initialized")
    
    classifier = OCRclassifier('./classification/letter_level/')
    print("Classifier loaded")
    
    # Text converter without correction
    converter_no_correction = textconverter(
        dir_to_xlsx='./utils/',
        use_lexical_correction=False
    )
    print("Text converter (without correction) initialized")
    
    # Text converter with correction
    converter_with_correction = textconverter(
        dir_to_xlsx='./utils/',
        use_lexical_correction=True,
        dictionary_path='./Dictionary/dictall.txt',
        max_distance=2
    )
    print("Text converter (with correction) initialized")
    
    # Process each test image
    print("\n" + "-"*70)
    print("Processing Test Images")
    print("-"*70)
    
    all_results = []
    
    for img_name, img_path in test_images:
        print(f"\nProcessing: {img_name}")
        
        # Optional: Preprocess image for degraded manuscripts
        # Uncomment the following lines to enable preprocessing:
        # import cv2
        # img = cv2.imread(img_path)
        # preprocessed = preprocessor.preprocess(img, denoise_strength=10)
        # temp_path = img_path.replace('.jpg', '_preprocessed.jpg')
        # cv2.imwrite(temp_path, preprocessed)
        # img_path = temp_path
        
        # Step 1: Segment image
        print("  1. Segmenting image...")
        segmenter = pagesegmenter(img_path)
        letter_array = segmenter.get_letter_coordinates(two_column_layout=False)
        
        num_words = len(letter_array)
        num_letters = sum(len(word) for word in letter_array)
        print(f"     Found {num_words} words, {num_letters} letters")
        
        # Step 2: Classify letters
        print("  2. Classifying letters...")
        x_test = segmenter.get_letters_for_classification(letter_array, './tmp_letters', cleanup=False)
        result_itrans = classifier.classify(x_test)
        print(f"     Classified {len(result_itrans)} letters")
        
        # Step 3a: Convert to words WITHOUT correction
        print("  3a. Converting to words (WITHOUT correction)...")
        words_no_correction = converter_no_correction.letterstoword(letter_array, result_itrans.copy())
        print(f"     Generated {len(words_no_correction)} words")
        
        # Step 3b: Convert to words WITH correction
        print("  3b. Converting to words (WITH correction)...")
        words_with_correction = converter_with_correction.letterstoword(letter_array, result_itrans.copy())
        print(f"     Generated {len(words_with_correction)} words (with correction)")
        
        # Get ground truth
        gt_words = ground_truth[img_name]
        
        # Calculate WER
        wer_no_correction, errors_no_correction, acc_no_correction = calculate_wer(gt_words, words_no_correction)
        wer_with_correction, errors_with_correction, acc_with_correction = calculate_wer(gt_words, words_with_correction)
        
        # Store results
        result = {
            'image': img_name,
            'ground_truth': gt_words,
            'ocr_no_correction': words_no_correction,
            'ocr_with_correction': words_with_correction,
            'wer_no_correction': wer_no_correction,
            'wer_with_correction': wer_with_correction,
            'errors_no_correction': errors_no_correction,
            'errors_with_correction': errors_with_correction,
            'accuracy_no_correction': acc_no_correction,
            'accuracy_with_correction': acc_with_correction
        }
        all_results.append(result)
        
        print(f"\n  Ground Truth: {gt_words[:5]}...")
        print(f"  OCR WITHOUT:  {words_no_correction[:5]}...")
        print(f"  OCR WITH:     {words_with_correction[:5]}...")
        
        print(f"\n  Results for {img_name}:")
        print(f"    WITHOUT Correction: WER={wer_no_correction:.2f}%, Accuracy={acc_no_correction:.2f}%")
        print(f"    WITH Correction:    WER={wer_with_correction:.2f}%, Accuracy={acc_with_correction:.2f}%")
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    
    total_words = sum(len(r['ocr_no_correction']) for r in all_results)
    print(f"\n   Total words extracted: {total_words}")
    print(f"   Images processed: {len(all_results)}")
    
    # Calculate aggregate WER
    print("\n" + "-"*70)
    print("WER CALCULATION (With Ground Truth)")
    print("-"*70)
    
    # Aggregate metrics
    total_gt_words = sum(len(r['ground_truth']) for r in all_results)
    total_errors_no_correction = sum(r['errors_no_correction'] for r in all_results)
    total_errors_with_correction = sum(r['errors_with_correction'] for r in all_results)
    
    avg_wer_no_correction = (total_errors_no_correction / total_gt_words) * 100 if total_gt_words > 0 else 0
    avg_wer_with_correction = (total_errors_with_correction / total_gt_words) * 100 if total_gt_words > 0 else 0
    
    avg_acc_no_correction = max(0, 100 - avg_wer_no_correction)
    avg_acc_with_correction = max(0, 100 - avg_wer_with_correction)
    
    correct_no_correction = total_gt_words - total_errors_no_correction
    correct_with_correction = total_gt_words - total_errors_with_correction
    
    print(f"\n   WITHOUT Lexical Correction:")
    print(f"      WER: {avg_wer_no_correction:.2f}%")
    print(f"      Accuracy: {avg_acc_no_correction:.2f}%")
    print(f"      Correct: {correct_no_correction}/{total_gt_words}")
    print(f"      Errors: {total_errors_no_correction}")
    
    print(f"\n   WITH Lexical Correction:")
    print(f"      WER: {avg_wer_with_correction:.2f}%")
    print(f"      Accuracy: {avg_acc_with_correction:.2f}%")
    print(f"      Correct: {correct_with_correction}/{total_gt_words}")
    print(f"      Errors: {total_errors_with_correction}")
    
    # Improvement metrics
    print("\n" + "="*70)
    print("IMPROVEMENT METRICS")
    print("="*70)
    wer_reduction = avg_wer_no_correction - avg_wer_with_correction
    relative_reduction = (wer_reduction / avg_wer_no_correction * 100) if avg_wer_no_correction > 0 else 0
    acc_improvement = avg_acc_with_correction - avg_acc_no_correction
    errors_fixed = total_errors_no_correction - total_errors_with_correction
    
    print(f"   WER Reduction: {wer_reduction:.2f} percentage points")
    print(f"   Relative WER Reduction: {relative_reduction:.2f}%")
    print(f"   Accuracy Improvement: {acc_improvement:.2f} percentage points")
    print(f"   Additional Errors Fixed: {errors_fixed}")
    
    # Detailed comparison
    print("\n" + "="*70)
    print("DETAILED WORD COMPARISON (First 10)")
    print("="*70)
    
    for result in all_results:
        gt = result['ground_truth']
        no_corr = result['ocr_no_correction']
        with_corr = result['ocr_with_correction']
        
        for i in range(min(10, len(gt))):
            gt_word = gt[i] if i < len(gt) else ''
            no_corr_word = no_corr[i] if i < len(no_corr) else ''
            with_corr_word = with_corr[i] if i < len(with_corr) else ''
            
            if gt_word == with_corr_word:
                status = "CORRECT"
            else:
                status = "WRONG"
            
            print(f"   {status:12} [{i}]")
            print(f"      GT:      '{gt_word}'")
            print(f"      WITHOUT: '{no_corr_word}'")
            print(f"      WITH:    '{with_corr_word}'")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETED")
    print("="*70)


if __name__ == "__main__":
    evaluate_on_test_images()
