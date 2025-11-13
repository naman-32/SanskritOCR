# Bhagavad Gita Verse Extraction from ITRANS Source
# Downloads from sanskritdocuments.org and parses into ground truth format
# Author: Naman Goenka
# Date: Dec 25th 2020

import os
import re
import urllib.request


def download_gita(output_file='bhagavad_gita.itx'):
    """Download complete Bhagavad Gita in ITRANS format."""
    url = "https://sanskritdocuments.org/doc_giitaa/bhagvadnew.itx"
    
    print(f"Downloading from {url}")
    
    req = urllib.request.Request(
        url,
        headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'}
    )
    
    with urllib.request.urlopen(req) as response:
        content = response.read()
    
    with open(output_file, 'wb') as f:
        f.write(content)
    
    print(f"Saved to {output_file}")
    return output_file


def parse_verses(filename):
    """
    Parse ITRANS file and extract verses.
    Returns list of dicts with verse_num and words.
    """
    verses = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check for verse number marker
        verse_num_match = re.search(r'\|\|\s*(\d+\\-\d+)\s*\|\|', line)
        
        if verse_num_match:
            verse_num = verse_num_match.group(1).replace('\\-', '-')
            verse_text = re.sub(r'\|\|\s*\d+\\-\d+\s*\|\|.*$', '', line)
            
            # Look back up to 3 lines for full verse content
            verse_lines = []
            for lookback in range(1, 4):
                if i >= lookback:
                    prev_line = lines[i-lookback].strip()
                    if (not prev_line.startswith('%') and
                        not re.search(r'\|\|\s*\d+\\-\d+\s*\|\|', prev_line) and
                        prev_line):
                        verse_lines.append(prev_line)
                        if prev_line.endswith('uvAcha |'):
                            break
            
            verse_lines.reverse()
            verse_lines.append(verse_text)
            
            full_text = ' '.join(verse_lines)
            full_text = full_text.replace('|', ' ')
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            full_text = re.sub(r'\d+\\-\d+', '', full_text)
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            words = [w for w in full_text.split() if w]
            
            if words:
                verses.append({
                    'verse_num': verse_num,
                    'words': words
                })
        
        i += 1
    
    return verses


def create_ground_truth(verses, output_file='ground_truth.txt', max_verses=None):
    """Create ground truth file in CSV format."""
    if max_verses:
        verses = verses[:max_verses]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Bhagavad Gita Ground Truth\n")
        f.write("# Format: image_filename,word1,word2,word3,...\n\n")
        
        for verse in verses:
            verse_num = verse['verse_num']
            words = verse['words']
            img_name = f"gita_{verse_num.replace('-', '_')}.jpg"
            f.write(f"{img_name},{','.join(words)}\n")
    
    print(f"Created ground truth with {len(verses)} verses")
    return len(verses)


def main():
    """Main extraction workflow."""
    print("Bhagavad Gita Verse Extraction")
    print("-" * 50)
    
    # Clean up existing files
    if os.path.exists('ground_truth.txt'):
        os.remove('ground_truth.txt')
        print("Cleaned up existing ground_truth.txt")
    
    # Download
    itx_file = download_gita()
    
    # Parse
    print("\nParsing verses...")
    verses = parse_verses(itx_file)
    print(f"Found {len(verses)} verses")
    
    # Create ground truth for first 50 verses
    print("\nCreating ground truth...")
    create_ground_truth(verses, 'ground_truth.txt', max_verses=50)
    
    # Show sample
    print("\nSample verses:")
    for i, verse in enumerate(verses[:3]):
        print(f"  {verse['verse_num']}: {' '.join(verse['words'][:5])}...")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
