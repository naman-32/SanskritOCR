# Complete Pipeline for Gita Dataset Generation
# Runs extraction, rendering, and dataset generation in one go
# Author: Naman Goenka
# Date: Dec 25th 2020

import sys
import os
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import individual modules
from extract_verses import download_gita, parse_verses, create_ground_truth
from render_images import load_ground_truth, itrans_to_devanagari, render_verse
from generate_dataset import DatasetGenerator


def cleanup():
    """Clean up existing files and directories."""
    print("Cleaning up existing files...")
    
    items_to_remove = [
        'ground_truth.txt',
        'bhagavad_gita.itx',
        'sample_images',
        'sample_dataset',
        '../tmp_words',
        '../tmp_letters'
    ]
    
    for item in items_to_remove:
        if os.path.exists(item):
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
            print(f"  Removed {item}")


def step1_extract_verses(max_verses=10):
    """Step 1: Extract verses and create ground truth."""
    print("\n" + "="*60)
    print("STEP 1: Extract Verses")
    print("="*60)
    
    # Download
    print("\nDownloading ITRANS source...")
    itx_file = download_gita()
    
    # Parse
    print("Parsing verses...")
    verses = parse_verses(itx_file)
    print(f"Found {len(verses)} verses")
    
    # Create ground truth
    print(f"Creating ground truth for first {max_verses} verses...")
    create_ground_truth(verses, 'ground_truth.txt', max_verses=max_verses)
    
    print(f"\nStep 1 complete: {max_verses} verses extracted")
    return max_verses


def step2_render_images(max_images=10):
    """Step 2: Render verse images."""
    print("\n" + "="*60)
    print("STEP 2: Render Images")
    print("="*60)
    
    # Load ground truth
    ground_truth = load_ground_truth('ground_truth.txt')
    print(f"Loaded {len(ground_truth)} verses")
    
    # Create output directory
    os.makedirs('sample_images', exist_ok=True)
    
    # Render images
    print(f"Rendering first {max_images} images...")
    count = 0
    for img_name, words in list(ground_truth.items())[:max_images]:
        itrans_text = ' '.join(words)
        devanagari_text = itrans_to_devanagari(itrans_text)
        
        output_path = os.path.join('sample_images', img_name)
        render_verse(devanagari_text, output_path)
        
        count += 1
        print(f"  {count}. {img_name}")
    
    print(f"\nStep 2 complete: {count} images rendered")
    return count


def step3_generate_dataset(max_verses=10):
    """Step 3: Generate labeled letter dataset."""
    print("\n" + "="*60)
    print("STEP 3: Generate Dataset")
    print("="*60)
    
    generator = DatasetGenerator()
    
    print(f"Generating dataset from {max_verses} verses...")
    total_letters = generator.generate(
        ground_truth_file='ground_truth.txt',
        images_dir='sample_images',
        output_dir='sample_dataset',
        max_verses=max_verses
    )
    
    print(f"\nStep 3 complete: {total_letters} letters saved")
    return total_letters


def verify_dataset():
    """Verify the generated dataset."""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    # Count classes and samples
    if not os.path.exists('sample_dataset'):
        print("ERROR: sample_dataset directory not found!")
        return False
    
    classes = [d for d in os.listdir('sample_dataset') if d.startswith('class_')]
    total_samples = 0
    
    print(f"\nDataset structure:")
    print(f"  Total classes: {len(classes)}")
    
    # Show top 10 classes by sample count
    class_counts = []
    for class_dir in classes:
        class_path = os.path.join('sample_dataset', class_dir)
        if os.path.isdir(class_path):
            count = len([f for f in os.listdir(class_path) if f.endswith('.png')])
            total_samples += count
            class_counts.append((class_dir, count))
    
    class_counts.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  Total samples: {total_samples}")
    print(f"\nTop 10 classes by sample count:")
    for i, (class_name, count) in enumerate(class_counts[:10], 1):
        print(f"    {i:2d}. {class_name:30s}: {count:3d} samples")
    
    # Verify sample images exist
    print(f"\nSample images:")
    if os.path.exists('sample_images'):
        images = [f for f in os.listdir('sample_images') if f.endswith('.jpg')]
        print(f"  Found {len(images)} images")
        for img in sorted(images)[:5]:
            print(f"    - {img}")
        if len(images) > 5:
            print(f"    ... and {len(images) - 5} more")
    
    return True


def main():
    """Run complete pipeline."""
    print("="*60)
    print("GITA DATASET AUGMENTATION PIPELINE")
    print("="*60)
    print("\nThis script will:")
    print("  1. Extract verses from ITRANS source")
    print("  2. Render verse images in Devanagari")
    print("  3. Generate labeled letter dataset")
    print()
    
    # Configuration
    NUM_VERSES = 10
    
    try:
        # Cleanup
        cleanup()
        
        # Run pipeline
        step1_extract_verses(max_verses=NUM_VERSES)
        step2_render_images(max_images=NUM_VERSES)
        step3_generate_dataset(max_verses=NUM_VERSES)
        
        # Verify
        verify_dataset()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print("\nOutput:")
        print("  - ground_truth.txt: ITRANS ground truth")
        print("  - sample_images/: Rendered verse images")
        print("  - sample_dataset/: Labeled letter dataset")
        print("\nDataset ready for training!")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
