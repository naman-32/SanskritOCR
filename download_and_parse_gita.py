# Download and Parse Bhagavad Gita ITRANS File
# Downloads from sanskritdocuments.org and extracts verses
# Author: Naman Goenka
# Date: Dec 25th 2020

import os
import re
import urllib.request

def download_complete_gita(output_dir='./verses/gita_itrans'):
    """
    Download complete Bhagavad Gita in ITRANS format (all 700 verses).
    
    Args:
        output_dir: Directory to save downloaded file
    
    Returns:
        Path to downloaded file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    url = "https://sanskritdocuments.org/doc_giitaa/bhagvadnew.itx"
    output_file = os.path.join(output_dir, "bhagavad_gita_complete.itx")
    
    print(f"📥 Downloading Complete Bhagavad Gita (all 700 verses)...")
    print(f"   URL: {url}")
    
    try:
        # Add user agent header to avoid 406 error
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
        )
        
        with urllib.request.urlopen(req) as response:
            content = response.read()
        
        with open(output_file, 'wb') as f:
            f.write(content)
        
        print(f"   ✓ Saved to: {output_file}")
        return output_file
    except Exception as e:
        print(f"   ✗ Error downloading: {e}")
        return None


def parse_itrans_file(filename):
    """
    Parse ITRANS file and extract verses.
    
    Verse format can be:
    1. Single line: verse_text || verse_number ||
    2. Multi-line with speaker:
       speaker uvAcha |
       verse_line1 |
       verse_line2 || verse_number ||
    
    Args:
        filename: Path to .itx file
    
    Returns:
        List of verse dictionaries with verse_num and words
    """
    verses = []
    
    print(f"\n📖 Parsing: {filename}")
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Process line by line to capture multi-line verses
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check if this line contains a verse number marker
            verse_num_match = re.search(r'\|\|\s*(\d+\\-\d+)\s*\|\|', line)
            
            if verse_num_match:
                verse_num = verse_num_match.group(1).replace('\\-', '-')
                
                # Extract verse text from this line (before the verse number)
                verse_text = re.sub(r'\|\|\s*\d+\\-\d+\s*\|\|.*$', '', line)
                
                # Look back 1-3 lines to capture full verse INCLUDING speaker
                verse_lines = []
                
                # Check up to 3 lines back to find speaker and verse content
                for lookback in range(1, 4):
                    if i >= lookback:
                        prev_line = lines[i-lookback].strip()
                        # Skip comments and lines with verse numbers
                        if (not prev_line.startswith('%') and
                            not re.search(r'\|\|\s*\d+\\-\d+\s*\|\|', prev_line) and
                            prev_line):  # Not empty
                            verse_lines.append(prev_line)
                            
                            # If this is a speaker line, we've found the start
                            if prev_line.endswith('uvAcha |'):
                                break
                
                # Reverse to get correct order
                verse_lines.reverse()
                
                # Add current line text
                verse_lines.append(verse_text)
                
                # Combine all lines
                full_verse_text = ' '.join(verse_lines)
                
                # Clean up
                full_verse_text = full_verse_text.replace('|', ' ')
                full_verse_text = re.sub(r'\s+', ' ', full_verse_text).strip()
                
                # Remove any remaining verse number patterns (like "1\-2")
                full_verse_text = re.sub(r'\d+\\-\d+', '', full_verse_text)
                full_verse_text = re.sub(r'\s+', ' ', full_verse_text).strip()
                
                # Split into words
                words = full_verse_text.split()
                
                # Filter out empty words
                words = [w for w in words if w]
                
                if words:
                    verses.append({
                        'verse_num': verse_num,
                        'words': words,
                        'text': full_verse_text
                    })
            
            i += 1
        
        print(f"   ✓ Found {len(verses)} verses")
        return verses
        
    except Exception as e:
        print(f"   ✗ Error parsing file: {e}")
        return []


def create_ground_truth_file(verses, chapter_num, output_file='./verses/gita_ground_truth.txt'):
    """
    Create ground truth file for dataset generation.
    
    Args:
        verses: List of verse dictionaries
        chapter_num: Chapter number
        output_file: Path to output ground truth file
    
    Returns:
        Number of verses written
    """
    print(f"\n📝 Creating ground truth file: {output_file}")
    
    # Read existing content if file exists
    existing_lines = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_lines = f.readlines()
    
    # Append new verses
    with open(output_file, 'a', encoding='utf-8') as f:
        # Add header if file is new
        if not existing_lines:
            f.write("# Ground Truth for Bhagavad Gita Verses\n")
            f.write("# Format: image_filename,word1,word2,word3,...\n")
            f.write("# Generated from sanskritdocuments.org ITRANS files\n\n")
        
        # Write verses
        for verse in verses:
            verse_num = verse['verse_num']
            words = verse['words']
            
            # Create image filename: gita_1_1.jpg for verse 1-1
            img_name = f"gita_{verse_num.replace('-', '_')}.jpg"
            
            # Write ground truth line
            line = f"{img_name},{','.join(words)}\n"
            f.write(line)
    
    print(f"   ✓ Added {len(verses)} verses to ground truth file")
    return len(verses)


def display_sample_verses(verses, num_samples=5):
    """Display sample verses for verification."""
    print(f"\n{'='*70}")
    print(f"📋 SAMPLE VERSES (First {num_samples})")
    print(f"{'='*70}")
    
    for i, verse in enumerate(verses[:num_samples]):
        print(f"\nVerse {verse['verse_num']}:")
        print(f"  Words: {verse['words']}")
        print(f"  Text: {verse['text']}")


def main():
    """Main function to download and parse Bhagavad Gita."""
    
    print("="*70)
    print("🕉️  BHAGAVAD GITA ITRANS DOWNLOADER & PARSER")
    print("="*70)
    print("\nThis script downloads complete Bhagavad Gita in ITRANS format")
    print("and creates ground truth files for dataset generation.\n")
    
    # Configuration
    OUTPUT_DIR = './verses/gita_itrans'
    GROUND_TRUTH_FILE = './verses/gita_ground_truth.txt'
    MAX_VERSES = 10  # Limit to first 10 verses for testing (set to None for all 700)
    
    print(f"📂 Output directory: {OUTPUT_DIR}")
    print(f"📝 Ground truth file: {GROUND_TRUTH_FILE}")
    if MAX_VERSES:
        print(f"📖 Processing first {MAX_VERSES} verses (for testing)")
    else:
        print(f"📖 Processing all 700 verses")
    
    # Download complete Gita
    print(f"\n{'='*70}")
    print(f"Downloading Complete Bhagavad Gita")
    print(f"{'='*70}")
    
    itx_file = download_complete_gita(OUTPUT_DIR)
    
    if not itx_file:
        print(f"✗ Download failed. Exiting.")
        return
    
    # Parse verses
    print(f"\n{'='*70}")
    print(f"Parsing Verses")
    print(f"{'='*70}")
    
    verses = parse_itrans_file(itx_file)
    
    if not verses:
        print(f"✗ No verses found. Exiting.")
        return
    
    # Limit verses if specified
    if MAX_VERSES:
        verses = verses[:MAX_VERSES]
        print(f"   ℹ️  Limited to first {len(verses)} verses for testing")
    
    # Create ground truth
    print(f"\n{'='*70}")
    print(f"Creating Ground Truth File")
    print(f"{'='*70}")
    
    # Clear existing file if starting fresh
    if os.path.exists(GROUND_TRUTH_FILE):
        os.remove(GROUND_TRUTH_FILE)
        print(f"   ℹ️  Removed existing ground truth file")
    
    num_written = create_ground_truth_file(verses, 0, GROUND_TRUTH_FILE)
    
    # Display samples
    display_sample_verses(verses, num_samples=5)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"✅ DOWNLOAD & PARSING COMPLETE")
    print(f"{'='*70}")
    print(f"\n📊 Summary:")
    print(f"   Total verses processed: {num_written}")
    print(f"   Ground truth file: {GROUND_TRUTH_FILE}")
    print(f"   ITRANS source file: {itx_file}")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Review the ground truth file: {GROUND_TRUTH_FILE}")
    print(f"   2. Obtain/render verse images (gita_1_1.jpg, gita_1_2.jpg, etc.)")
    print(f"   3. Place images in ./verses/ directory")
    print(f"   4. Run: python generate_training_data.py")
    print(f"   5. Generate training data from verses!")
    
    print(f"\n📚 To process more verses:")
    print(f"   Edit MAX_VERSES in this script")
    print(f"   Set MAX_VERSES = None to process all 700 verses")
    print(f"   Set MAX_VERSES = 50 to process first 50 verses, etc.")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
