# Bhagavad Gita Verse Image Renderer
# Converts ITRANS to Devanagari and generates image files
# Author: Naman Goenka
# Date: Dec 25th 2020

import os
from PIL import Image, ImageDraw, ImageFont
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


def itrans_to_devanagari(itrans_text):
    """Convert ITRANS text to Devanagari script."""
    itrans_text = itrans_text.replace('^', '')
    return transliterate(itrans_text, sanscript.ITRANS, sanscript.DEVANAGARI)


def render_verse(text, output_path, font_size=48, width=1200):
    """Render verse text as image."""
    # Try to find Devanagari font
    font_paths = [
        '/System/Library/Fonts/Supplemental/Devanagari Sangam MN.ttc',
        '/Library/Fonts/Kohinoor.ttc',
        '/System/Library/Fonts/Supplemental/DevanagariMT.ttc',
    ]
    
    font = None
    for path in font_paths:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, font_size)
                break
            except:
                continue
    
    if font is None:
        font = ImageFont.load_default()
    
    # Measure text and wrap lines
    temp_img = Image.new('RGB', (width, 1000), 'white')
    temp_draw = ImageDraw.Draw(temp_img)
    
    words = text.split()
    lines = []
    current_line = []
    padding = 50
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = temp_draw.textbbox((0, 0), test_line, font=font)
        if bbox[2] - bbox[0] <= width - 2 * padding:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Create final image
    line_height = font_size + 20
    height = len(lines) * line_height + 2 * padding
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill='black', font=font)
        y += line_height
    
    img.save(output_path)


def load_ground_truth(filename):
    """Load ground truth file."""
    ground_truth = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                ground_truth[parts[0]] = parts[1:]
    return ground_truth


def main():
    """Main rendering workflow."""
    print("Bhagavad Gita Image Rendering")
    print("-" * 50)
    
    # Clean up existing images
    import shutil
    if os.path.exists('sample_images'):
        shutil.rmtree('sample_images')
        print("Cleaned up existing sample_images/")
    
    # Load ground truth
    ground_truth = load_ground_truth('ground_truth.txt')
    print(f"Loaded {len(ground_truth)} verses")
    
    # Create output directory
    os.makedirs('sample_images', exist_ok=True)
    
    # Render first 10 verses
    count = 0
    for img_name, words in list(ground_truth.items())[:10]:
        itrans_text = ' '.join(words)
        devanagari_text = itrans_to_devanagari(itrans_text)
        
        output_path = os.path.join('sample_images', img_name)
        render_verse(devanagari_text, output_path)
        
        count += 1
        print(f"Rendered {img_name}")
    
    print(f"\nRendered {count} images to sample_images/")


if __name__ == "__main__":
    main()
