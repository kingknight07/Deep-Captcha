# deep_captcha_library.py

import os
import random
import string
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS


class DeepCaptcha:
    """
    A highly configurable, self-contained CAPTCHA generator library.
    Version 10.0: Added strike lines feature, removed wavy text and wavy lines.
    """

    def __init__(self,
                 width: int = 280,
                 height: int = 100,
                 text_length: int = 5,
                 num_lines: int = 8,
                 line_thickness: int = 3,
                 dot_radius: int = 0,
                 blur_level: float = 0.5,
                 shear_text: bool = True,
                 noise_density: float = 0.5,
                 char_colors: list = None):
        """
        Initializes the DeepCaptcha generator with user-defined settings.

        Args:
            width (int): Width of the CAPTCHA image in pixels. Default: 280.
            height (int): Height of the CAPTCHA image in pixels. Default: 100.
            text_length (int): Number of characters in the CAPTCHA text. Default: 5.
            num_lines (int): Number of strike lines to draw across the text. Default: 8.
            line_thickness (int): The width of the strike lines in pixels. Default: 3.
            dot_radius (int): Radius of background noise dots. Default: 0.
            blur_level (float): Intensity of blur effect (0.0 to 1.0). Default: 0.5.
            shear_text (bool): If True, characters are distorted with a shear effect. Default: True.
            noise_density (float): Density of background noise dots (0.0 to 1.0). Default: 0.5.
            char_colors (list): List of RGB tuples for character colors. Default: None (uses predefined colors).
        """
        self.width, self.height, self.text_length = width, height, text_length
        self.num_lines, self.line_thickness, self.dot_radius = num_lines, line_thickness, dot_radius
        self.shear_text = shear_text
        self.blur_radius = max(0, min(1.0, blur_level)) * 1.5
        max_dots = self.width * self.height * 0.1
        self.num_dots = int(max(0, min(1.0, noise_density)) * max_dots)

        if char_colors:
            self.char_colors = char_colors
        else:
            self.char_colors = [(180, 0, 0), (0, 150, 0), (0, 0, 180), (150, 0, 150), (139, 69, 19)]

        script_dir = os.path.dirname(os.path.abspath(__file__))
        font_dir = os.path.join(script_dir, 'static')
        if not os.path.isdir(font_dir): raise RuntimeError(f"Font directory not found: {font_dir}")
        allowed_weights = ['bold', 'black', 'extrabold', 'semibold']
        font_paths = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if
                      f.lower().endswith('.ttf') and any(w in f.lower() for w in allowed_weights)]
        if not font_paths: raise RuntimeError(f"No bold fonts found in: {font_dir}")
        self.fonts = [ImageFont.truetype(fp, fs) for fp in font_paths for fs in [42, 50, 56]]

    def _generate_random_text(self):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=self.text_length))

    def _draw_background_dots(self, draw):
        # ... (no changes needed in this method)
        dot_colors = [(180, 0, 0), (0, 180, 0), (0, 0, 180)]
        for _ in range(self.num_dots):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            bbox = [x - self.dot_radius, y - self.dot_radius, x + self.dot_radius, y + self.dot_radius]
            draw.ellipse(bbox, fill=random.choice(dot_colors)) if self.dot_radius > 0 else draw.point((x, y),
                                                                                                      fill=random.choice(
                                                                                                          dot_colors))

    def _draw_strike_lines(self, draw):
        """Draw simple strike lines across the captcha image."""
        line_colors = [(50, 50, 50), (0, 0, 0), (80, 80, 80)]
        
        for _ in range(self.num_lines):
            # Generate random start and end points for strike lines
            # Lines can be horizontal, diagonal, or slightly curved
            line_type = random.choice(['horizontal', 'diagonal', 'vertical'])
            
            if line_type == 'horizontal':
                # Horizontal lines across the image
                y = random.randint(self.height // 4, 3 * self.height // 4)
                start = (0, y)
                end = (self.width, y + random.randint(-10, 10))
            elif line_type == 'diagonal':
                # Diagonal lines from corner to corner variations
                start = (random.randint(0, self.width // 3), random.randint(0, self.height))
                end = (random.randint(2 * self.width // 3, self.width), random.randint(0, self.height))
            else:  # vertical
                # Vertical or near-vertical lines
                x = random.randint(self.width // 4, 3 * self.width // 4)
                start = (x, 0)
                end = (x + random.randint(-10, 10), self.height)
            
            draw.line([start, end], fill=random.choice(line_colors), width=self.line_thickness)

    def generate(self) -> tuple[Image.Image, str]:
        image = Image.new('RGB', (self.width, self.height), 'white')
        draw = ImageDraw.Draw(image)

        # Step 1: Draw background dots.
        if self.num_dots > 0: self._draw_background_dots(draw)

        # Step 2: Create the text block completely separately.
        text = self._generate_random_text()
        buffer = Image.new('RGBA', (self.width * 3, self.height * 2), (0, 0, 0, 0))
        x_pos, overlap = 20, 20
        for char in text:
            font = random.choice(self.fonts)
            color = random.choice(self.char_colors)
            char_img = Image.new('RGBA', (120, 120), (0, 0, 0, 0))
            ImageDraw.Draw(char_img).text((10, 10), char, font=font, fill=color)

            # --- NEW: Apply shear transformation if enabled ---
            if self.shear_text:
                shear_factor = random.uniform(-0.4, 0.4)
                # Affine transform: (a, b, c, d, e, f) where b is horizontal shear
                char_img = char_img.transform(char_img.size, Image.AFFINE, (1, shear_factor, 0, 0, 1, 0))

            # No rotation applied - wavy_text removed
            rotated_char = char_img
            buffer.paste(rotated_char, (x_pos, random.randint(30, 50)), rotated_char)
            x_pos += rotated_char.width - overlap

        text_bbox = buffer.getbbox()
        if text_bbox:
            text_block = buffer.crop(text_bbox)
            if text_block.width > self.width:
                new_width = self.width
                new_height = int(text_block.height * (new_width / text_block.width))
                text_block = text_block.resize((new_width, new_height), resample=LANCZOS)

            # Step 3: Paste the text block onto the main image.
            paste_x = (self.width - text_block.width) // 2
            paste_y = (self.height - text_block.height) // 2
            image.paste(text_block, (paste_x, paste_y), text_block)

        # --- Draw strike lines over the text ---
        if self.num_lines > 0: 
            self._draw_strike_lines(draw)

        # Step 5: Apply the final blur to the composite image.
        if self.blur_radius > 0: image = image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        return image, text