# Render Bhagavad Gita Verses as Images
# Converts ITRANS to Devanagari and renders as images
# Author: Naman Goenka
# Date: Dec 25th 2020

import os
from PIL import Image, ImageDraw, ImageFont
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def itrans_to_devanagari(itrans_text):
    """
    Convert ITRANS text to Devanagari.
    
    Args:
        itrans_text: Text in ITRANS format
    
    Returns:
        Text in Devanagari script
    """
    try:
        # Handle special ITRANS characters
        # Replace ^ with empty string (it's used for vowel length in some schemes)
        itrans_text = itrans_text.replace('^', '')
        
        # Transliterate
        devanagari = transliterate(itrans_text, sanscript.ITRANS, sanscript.DEVANAGARI)
        return devanagari
    except Exception as e:
        print(f"   ⚠️  Error transliterating '{itrans_text}': {e}")
        return itrans_text


def render_verse_image(verse_text, output_path, 
                       font_size=48, 
                       image_width=1200, 
                       line_spacing=20,
                       padding=50):
    """
    Render a verse as an image.
    
    Args:
        verse_text: Text in Devanagari
        output_path: Path to save the image
        font_size: Font size for rendering
        image_width: Width of the image
        line_spacing: Space between lines
        padding: Padding around text
    """
    try:
        # Try to use a Sanskrit/Devanagari font
        # Common Sanskrit fonts on macOS
        font_paths = [
            '/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc',
            '/Library/Fonts/Kohinoor.ttc',
            '/System/Library/Fonts/Supplemental/DevanagariMT.ttc',
        ]
        
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    print(f"   ✓ Using font: {os.path.basename(font_path)}")
                    break
                except:
                    continue
        
        if font is None:
            print(f"   ⚠️  No Devanagari font found, using default")
            font = ImageFont.load_default()
        
        # Create a temporary image to measure text size
        temp_img = Image.new('RGB', (image_width, 1000), color='white')
        temp_draw = ImageDraw.Draw(temp_img)
        
        # Split text into words and wrap lines
        words = verse_text.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            bbox = temp_draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            
            if text_width <= image_width - 2 * padding:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Calculate image height
        line_height = font_size + line_spacing
        image_height = len(lines) * line_height + 2 * padding
        
        # Create final image
        img = Image.new('RGB', (image_width, image_height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw text
        y = padding
        for line in lines:
            draw.text((padding, y), line, fill='black', font=font)
            y += line_height
        
        # Save image
        img.save(output_path)
        print(f"   ✓ Saved: {output_path}")
        
    except Exception as e:
        print(f"   ✗ Error rendering image: {e}")


def load_ground_truth(ground_truth_file):
    """
    Load ground truth file.
    
    Returns:
        Dictionary mapping image filename to ITRANS words
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
        print(f"✗ Ground truth file not found: {ground_truth_file}")
        return {}


def main():
    """Main function to render verse images."""
    
    print("="*70)
    print("🖼️  BHAGAVAD GITA VERSE IMAGE RENDERER")
    print("="*70)
    print("\nThis script converts ITRANS verses to Devanagari")
    print("and renders them as images for OCR training.\n")
    
    # Configuration
    GROUND_TRUTH_FILE = './verses/gita_ground_truth.txt'
    OUTPUT_DIR = './verses/'
    FONT_SIZE = 48  # Adjust for different text sizes
    IMAGE_WIDTH = 1200
    
    print(f"📂 Ground truth file: {GROUND_TRUTH_FILE}")
    print(f"📂 Output directory: {OUTPUT_DIR}")
    print(f"📝 Font size: {FONT_SIZE}")
    print(f"📐 Image width: {IMAGE_WIDTH}px")
    
    # Check if indic-transliteration is installed
    try:
        from indic_transliteration import sanscript
        print(f"\n✓ indic-transliteration library found")
    except ImportError:
        print(f"\n✗ Error: indic-transliteration library not found")
        print(f"   Please install: pip install indic-transliteration")
        return
    
    # Load ground truth
    print(f"\n{'='*70}")
    print(f"Loading Ground Truth")
    print(f"{'='*70}")
    
    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
    print(f"✓ Loaded {len(ground_truth)} verses")
    
    if not ground_truth:
        print("✗ No verses to render. Exiting.")
        return
    
    # Render each verse
    print(f"\n{'='*70}")
    print(f"Rendering Verse Images")
    print(f"{'='*70}")
    
    rendered_count = 0
    error_count = 0
    
    for image_name, itrans_words in ground_truth.items():
        print(f"\n📝 Processing: {image_name}")
        
        # Convert ITRANS words to sentence
        itrans_sentence = ' '.join(itrans_words)
        print(f"   ITRANS: {itrans_sentence[:80]}...")
        
        # Convert to Devanagari
        devanagari_sentence = itrans_to_devanagari(itrans_sentence)
        print(f"   Devanagari: {devanagari_sentence[:80]}...")
        
        # Render image
        output_path = os.path.join(OUTPUT_DIR, image_name)
        render_verse_image(devanagari_sentence, output_path, 
                          font_size=FONT_SIZE, 
                          image_width=IMAGE_WIDTH)
        
        rendered_count += 1
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ RENDERING COMPLETE")
    print(f"{'='*70}")
    print(f"\n📊 Summary:")
    print(f"   Verses rendered: {rendered_count}")
    print(f"   Output directory: {OUTPUT_DIR}")
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Review the generated images in: {OUTPUT_DIR}")
    print(f"   2. Run: ocrvenv/bin/python generate_training_data.py")
    print(f"   3. Generate training data from verse images!")
    
    print(f"\n📚 To adjust rendering:")
    print(f"   Edit FONT_SIZE (currently {FONT_SIZE}) for larger/smaller text")
    print(f"   Edit IMAGE_WIDTH (currently {IMAGE_WIDTH}) for wider/narrower images")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
