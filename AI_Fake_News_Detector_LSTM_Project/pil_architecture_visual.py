"""
Create Instagram-optimized architecture diagram using PIL/Pillow
Better text rendering and more control for social media content
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_pipeline_architecture_pil(output_path: str = "pipeline_architecture_pil.png", square: bool = True):
    """
    Create a clean, readable pipeline architecture using PIL.
    Perfect for Instagram with large, bold text.
    """
    # Dimensions
    if square:
        width, height = 1080, 1080
    else:
        width, height = 1080, 1920
    
    # Colors (dark theme)
    bg_color = (26, 26, 26)  # #1a1a1a
    stage_colors = [
        (88, 166, 255),   # Blue - Input
        (57, 211, 83),    # Green - Clean
        (255, 166, 87),   # Orange - CNN
        (189, 147, 249),  # Purple - LSTM
        (255, 107, 107),  # Red - Classify
        (78, 205, 196),   # Cyan - Result
    ]
    white = (255, 255, 255)
    gray = (180, 180, 180)
    
    # Create image
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Try to use a better font (fallback to default if not available)
    try:
        title_font = ImageFont.truetype("arial.ttf", 70)
        stage_font = ImageFont.truetype("arialbd.ttf", 48)
        emoji_font = ImageFont.truetype("seguiemj.ttf", 60)
    except:
        # Fallback to default
        title_font = ImageFont.load_default()
        stage_font = ImageFont.load_default()
        emoji_font = ImageFont.load_default()
    
    # Title
    title = "🔄 PIPELINE"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((width - title_width) // 2, 50), title, fill=(88, 166, 255), font=title_font)
    
    # Pipeline stages
    stages = [
        ("📰", "INPUT\nHeadlines"),
        ("🔧", "CLEAN\n& Process"),
        ("🧠", "CNN\nPatterns"),
        ("🔁", "LSTM\nSequence"),
        ("🎯", "CLASSIFY"),
        ("✅", "RESULT\nReal/Fake"),
    ]
    
    # Calculate positions
    start_y = 220
    box_height = 120
    spacing = 20
    box_width = width - 200
    x_start = 100
    
    for i, ((emoji, text), color) in enumerate(zip(stages, stage_colors)):
        y = start_y + i * (box_height + spacing)
        
        # Draw rounded rectangle (box)
        draw.rounded_rectangle(
            [(x_start, y), (x_start + box_width, y + box_height)],
            radius=15,
            fill=(*color, 60),  # Semi-transparent
            outline=color,
            width=6
        )
        
        # Draw emoji (left side)
        emoji_x = x_start + 60
        emoji_y = y + (box_height - 60) // 2
        draw.text((emoji_x, emoji_y), emoji, fill=white, font=emoji_font)
        
        # Draw text (center)
        text_lines = text.split('\n')
        text_height_total = len(text_lines) * 50
        text_y_start = y + (box_height - text_height_total) // 2 + 10
        
        for j, line in enumerate(text_lines):
            line_bbox = draw.textbbox((0, 0), line, font=stage_font)
            line_width = line_bbox[2] - line_bbox[0]
            text_x = x_start + (box_width - line_width) // 2 + 20
            text_y = text_y_start + j * 55
            draw.text((text_x, text_y), line, fill=white, font=stage_font)
        
        # Draw arrow to next stage (except last)
        if i < len(stages) - 1:
            arrow_x = width // 2
            arrow_y_start = y + box_height + 5
            arrow_y_end = y + box_height + spacing - 5
            draw.line([(arrow_x, arrow_y_start), (arrow_x, arrow_y_end)], fill=gray, width=8)
            # Arrow head
            draw.polygon([
                (arrow_x, arrow_y_end),
                (arrow_x - 15, arrow_y_end - 20),
                (arrow_x + 15, arrow_y_end - 20)
            ], fill=gray)
    
    # Save
    img.save(output_path, quality=95)
    print(f"✅ Created: {output_path}")
    return output_path


def create_model_architecture_pil(output_path: str = "model_architecture_pil.png", square: bool = True):
    """
    Create a clean, readable model architecture using PIL.
    Shows layers with better typography.
    """
    # Dimensions
    if square:
        width, height = 1080, 1080
    else:
        width, height = 1080, 1920
    
    # Colors
    bg_color = (26, 26, 26)
    layer_colors = [
        (255, 107, 157),  # Pink - Input
        (196, 69, 105),   # Dark Pink - Embed
        (249, 127, 81),   # Orange - CNN
        (254, 202, 87),   # Yellow - Pool
        (72, 219, 251),   # Cyan - LSTM
        (10, 189, 227),   # Blue - Dense
        (0, 210, 211),    # Teal - Output
    ]
    white = (255, 255, 255)
    gray = (180, 180, 180)
    
    # Create image
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 70)
        layer_font = ImageFont.truetype("arialbd.ttf", 50)
        spec_font = ImageFont.truetype("arial.ttf", 36)
    except:
        title_font = ImageFont.load_default()
        layer_font = ImageFont.load_default()
        spec_font = ImageFont.load_default()
    
    # Title
    title = "🧠 MODEL"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((width - title_width) // 2, 50), title, fill=(0, 217, 255), font=title_font)
    
    # Layers
    layers = [
        ("INPUT", "20 words"),
        ("EMBED", "100-dim"),
        ("CNN", "64 filters"),
        ("POOL", "Downsample"),
        ("LSTM", "64 units"),
        ("DENSE", "32 units"),
        ("OUTPUT", "Real/Fake"),
    ]
    
    # Calculate positions
    start_y = 200
    box_height = 100
    spacing = 20
    box_width = width - 200
    x_start = 100
    
    for i, ((layer_name, specs), color) in enumerate(zip(layers, layer_colors)):
        y = start_y + i * (box_height + spacing)
        
        # Draw box
        draw.rounded_rectangle(
            [(x_start, y), (x_start + box_width, y + box_height)],
            radius=15,
            fill=(*color, 70),
            outline=color,
            width=6
        )
        
        # Layer name (left)
        name_x = x_start + 40
        name_y = y + (box_height - 50) // 2
        draw.text((name_x, name_y), layer_name, fill=color, font=layer_font)
        
        # Specs (right)
        spec_bbox = draw.textbbox((0, 0), specs, font=spec_font)
        spec_width = spec_bbox[2] - spec_bbox[0]
        spec_x = x_start + box_width - spec_width - 40
        spec_y = y + (box_height - 36) // 2 + 5
        draw.text((spec_x, spec_y), specs, fill=white, font=spec_font)
        
        # Arrow to next (except last)
        if i < len(layers) - 1:
            arrow_x = width // 2
            arrow_y_start = y + box_height + 5
            arrow_y_end = y + box_height + spacing - 5
            draw.line([(arrow_x, arrow_y_start), (arrow_x, arrow_y_end)], fill=(0, 217, 255), width=8)
            # Arrow head
            draw.polygon([
                (arrow_x, arrow_y_end),
                (arrow_x - 15, arrow_y_end - 20),
                (arrow_x + 15, arrow_y_end - 20)
            ], fill=(0, 217, 255))
    
    # Save
    img.save(output_path, quality=95)
    print(f"✅ Created: {output_path}")
    return output_path


if __name__ == "__main__":
    print("🎨 Creating PIL-based architecture diagrams...")
    print("=" * 50)
    
    # Create both diagrams
    create_pipeline_architecture_pil("pipeline_architecture_pil_square.png", square=True)
    create_model_architecture_pil("model_architecture_pil_square.png", square=True)
    
    print("\n✨ Done! Much more readable for Instagram!")
