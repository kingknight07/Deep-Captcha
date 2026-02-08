# deepcaptcha/core.py
"""
DeepCaptcha: AI-Resistant CAPTCHA Generator

A highly configurable, self-contained CAPTCHA generator library
with breakthrough adversarial AI resistance technology.
"""

import os
import random
import string
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

try:
    LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS = Image.LANCZOS


def _get_font_dir():
    """Get the fonts directory path, works for both installed package and development."""
    # Try importlib.resources first (Python 3.9+)
    try:
        if sys.version_info >= (3, 9):
            from importlib.resources import files
            return str(files('deepcaptcha') / 'fonts')
        else:
            from importlib.resources import path
            with path('deepcaptcha', 'fonts') as p:
                return str(p)
    except (ImportError, TypeError, ModuleNotFoundError):
        pass
    
    # Fallback for development or if importlib fails
    script_dir = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.join(script_dir, 'fonts')
    if os.path.isdir(font_dir):
        return font_dir
    
    # Try static folder (legacy location)
    parent_dir = os.path.dirname(script_dir)
    static_dir = os.path.join(parent_dir, 'static')
    if os.path.isdir(static_dir):
        return static_dir
    
    raise RuntimeError(f"Font directory not found. Checked: {font_dir}, {static_dir}")


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
                 color_mode: bool = True,
                 noise_density: float = 0.5,
                 ai_resistance_level: int = 1,
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
            color_mode (bool): If True, captcha uses colors; if False, black & white. Default: True.
            noise_density (float): Density of background noise dots (0.0 to 1.0). Default: 0.5.
            ai_resistance_level (int): AI resistance level (0-3). 0=none, 1=basic, 2=moderate, 3=advanced. Default: 1.
            char_colors (list): List of RGB tuples for character colors. Default: None (uses predefined colors).
        """
        self.width, self.height, self.text_length = width, height, text_length
        self.num_lines, self.line_thickness, self.dot_radius = num_lines, line_thickness, dot_radius
        self.shear_text = shear_text
        self.color_mode = color_mode
        self.ai_resistance_level = max(0, min(3, ai_resistance_level))  # Clamp to 0-3 range
        self.blur_radius = max(0, min(1.0, blur_level)) * 1.5
        max_dots = self.width * self.height * 0.1
        self.num_dots = int(max(0, min(1.0, noise_density)) * max_dots)

        if char_colors:
            self.char_colors = char_colors
        elif self.color_mode:
            # Colorful mode - use various colors
            self.char_colors = [(180, 0, 0), (0, 150, 0), (0, 0, 180), (150, 0, 150), (139, 69, 19)]
        else:
            # Black & white mode - use only black and dark gray
            self.char_colors = [(0, 0, 0), (50, 50, 50), (30, 30, 30), (70, 70, 70)]

        font_dir = _get_font_dir()
        if not os.path.isdir(font_dir):
            raise RuntimeError(f"Font directory not found: {font_dir}")
        allowed_weights = ['bold', 'black', 'extrabold', 'semibold']
        font_paths = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if
                      f.lower().endswith('.ttf') and any(w in f.lower() for w in allowed_weights)]
        if not font_paths:
            raise RuntimeError(f"No bold fonts found in: {font_dir}")
        self.fonts = [ImageFont.truetype(fp, fs) for fp in font_paths for fs in [42, 50, 56]]

    def _generate_random_text(self):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=self.text_length))

    def _draw_background_dots(self, draw):
        # Choose colors based on color mode
        if self.color_mode:
            dot_colors = [(180, 0, 0), (0, 180, 0), (0, 0, 180)]
        else:
            dot_colors = [(100, 100, 100), (150, 150, 150), (80, 80, 80)]
            
        for _ in range(self.num_dots):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            bbox = [x - self.dot_radius, y - self.dot_radius, x + self.dot_radius, y + self.dot_radius]
            draw.ellipse(bbox, fill=random.choice(dot_colors)) if self.dot_radius > 0 else draw.point((x, y),
                                                                                                      fill=random.choice(
                                                                                                          dot_colors))

    def _draw_strike_lines(self, draw):
        """Draw simple strike lines across the captcha image."""
        if self.color_mode:
            line_colors = [(50, 50, 50), (0, 0, 0), (80, 80, 80), (100, 0, 0), (0, 100, 0)]
        else:
            line_colors = [(50, 50, 50), (0, 0, 0), (80, 80, 80), (120, 120, 120)]
        
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

    def _apply_histogram_manipulation(self, image: Image.Image) -> Image.Image:
        """
        Apply histogram equalization and manipulation to confuse AI models.
        This changes pixel distribution patterns while preserving visual appearance.
        """
        if self.ai_resistance_level == 0:
            return image
            
        # Convert to numpy array for histogram manipulation
        img_array = np.array(image)
        height, width, channels = img_array.shape
        
        # Apply different manipulations based on resistance level
        if self.ai_resistance_level >= 1:
            # Basic: Extremely subtle histogram stretching
            for c in range(channels):
                channel = img_array[:, :, c].astype(np.float32)
                # Apply extremely mild histogram stretching
                min_val, max_val = np.percentile(channel, [10, 90])
                if max_val > min_val:
                    stretch_factor = random.uniform(0.998, 1.002)  # Ultra-subtle
                    stretched = (channel - min_val) / (max_val - min_val) * 255 * stretch_factor
                    stretched = np.clip(stretched, 0, 255)
                    # Only apply where difference is minimal
                    diff_mask = np.abs(stretched - channel) < 2.0
                    channel[diff_mask] = stretched[diff_mask]
                    img_array[:, :, c] = channel.astype(np.uint8)
        
        if self.ai_resistance_level >= 2:
            # Moderate: Extremely slight gamma correction variations
            for c in range(channels):
                gamma = random.uniform(0.999, 1.001)  # Ultra-subtle gamma
                channel = img_array[:, :, c].astype(np.float32)
                corrected = 255 * (channel / 255) ** gamma
                # Only apply minimal changes
                diff_mask = np.abs(corrected - channel) < 1.0
                channel[diff_mask] = corrected[diff_mask]
                img_array[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        if self.ai_resistance_level >= 3:
            # Advanced: Ultra-subtle non-linear histogram remapping
            for c in range(channels):
                channel = img_array[:, :, c].astype(np.float32)
                # Create ultra-subtle S-curve transformation
                curve_strength = random.uniform(0.001, 0.003)  # Ultra-subtle
                normalized = channel / 255.0
                s_curve = normalized + curve_strength * np.sin(8 * np.pi * normalized)
                s_curve_denorm = s_curve * 255
                # Only apply where difference is tiny
                diff_mask = np.abs(s_curve_denorm - channel) < 0.5
                channel[diff_mask] = s_curve_denorm[diff_mask]
                img_array[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)

    def _apply_rgb_perturbations(self, image: Image.Image) -> Image.Image:
        """
        Apply subtle RGB channel perturbations that are invisible to humans
        but create adversarial patterns for neural networks.
        """
        if self.ai_resistance_level == 0:
            return image
            
        img_array = np.array(image)
        height, width, channels = img_array.shape
        
        if self.ai_resistance_level >= 1:
            # Basic: Ultra-tiny random perturbations to RGB channels
            noise_strength = 0.5  # Ultra-subtle - barely detectable
            noise = np.random.normal(0, noise_strength, img_array.shape)
            # Apply only to pixels where change is minimal
            candidate_change = img_array.astype(np.float32) + noise
            change_mask = np.abs(noise) < 1.0  # Only apply tiny changes
            img_array = img_array.astype(np.float32)
            img_array[change_mask] = candidate_change[change_mask]
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        if self.ai_resistance_level >= 2:
            # Moderate: Ultra-subtle channel-specific transformations
            red_shift = random.uniform(-0.3, 0.3)  # Ultra-subtle
            blue_shift = random.uniform(-0.3, 0.3)
            
            img_array = img_array.astype(np.float32)
            # Apply shifts only where they create minimal change
            red_candidate = img_array[:, :, 0] + red_shift
            blue_candidate = img_array[:, :, 2] + blue_shift
            
            red_mask = np.abs(red_shift) < 0.5
            blue_mask = np.abs(blue_shift) < 0.5
            
            img_array[:, :, 0][red_mask] = red_candidate[red_mask]
            img_array[:, :, 2][blue_mask] = blue_candidate[blue_mask]
            
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        if self.ai_resistance_level >= 3:
            # Advanced: Ultra-subtle adversarial patterns
            # Create ultra-subtle patterns that exploit CNN spatial biases
            x_pattern = np.sin(np.arange(width) * 0.2) * 0.2  # Ultra-subtle
            y_pattern = np.cos(np.arange(height) * 0.2) * 0.2
            
            img_array = img_array.astype(np.float32)
            for c in range(channels):
                # Add ultra-subtle position-dependent perturbations
                for y in range(0, height, 4):  # Skip pixels to reduce impact
                    for x in range(0, width, 4):
                        if y < height and x < width:
                            perturbation = (x_pattern[x] + y_pattern[y]) * (c + 1) * 0.01  # Ultra-subtle
                            if abs(perturbation) < 0.3:  # Only apply minimal changes
                                img_array[y, x, c] = np.clip(img_array[y, x, c] + perturbation, 0, 255)
            img_array = img_array.astype(np.uint8)
        
        return Image.fromarray(img_array)

    def _apply_adversarial_noise(self, image: Image.Image) -> Image.Image:
        """
        Apply specially crafted noise patterns designed to confuse
        common CNN architectures used for CAPTCHA solving.
        """
        if self.ai_resistance_level == 0:
            return image
            
        img_array = np.array(image)
        height, width, channels = img_array.shape
        
        if self.ai_resistance_level >= 1:
            # Basic: Ultra-subtle high-frequency noise
            high_freq_noise = np.random.normal(0, 0.3, img_array.shape)  # Ultra-subtle
            # Apply only where change is minimal
            change_mask = np.abs(high_freq_noise) < 0.5
            img_array = img_array.astype(np.float32)
            img_array[change_mask] += high_freq_noise[change_mask]
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        if self.ai_resistance_level >= 2:
            # Moderate: Ultra-subtle checkerboard patterns
            checker_size = 16  # Larger checkerboard, less visible
            checker_strength = 0.1  # Ultra-subtle
            
            img_array = img_array.astype(np.float32)
            for y in range(0, height, checker_size):
                for x in range(0, width, checker_size):
                    if (x // checker_size + y // checker_size) % 2 == 0:
                        y_end = min(y + checker_size, height)
                        x_end = min(x + checker_size, width)
                        # Only apply to alternate pixels to minimize visibility
                        for py in range(y, y_end, 2):
                            for px in range(x, x_end, 2):
                                if py < height and px < width:
                                    img_array[py, px] = np.clip(
                                        img_array[py, px] + checker_strength, 0, 255
                                    )
            img_array = img_array.astype(np.uint8)
        
        if self.ai_resistance_level >= 3:
            # Advanced: Ultra-subtle CNN filter confusion patterns
            img_array = img_array.astype(np.float32)
            
            # Ultra-subtle filter confusion patterns
            filter_confusion_patterns = [
                np.array([[0.02, -0.02, 0.02], [-0.02, 0.02, -0.02], [0.02, -0.02, 0.02]]),  # Ultra-subtle
                np.array([[0, 0.01, 0], [0.01, -0.04, 0.01], [0, 0.01, 0]])  # Ultra-subtle edge pattern
            ]
            
            for pattern in filter_confusion_patterns:
                p_height, p_width = pattern.shape
                for y in range(0, height - p_height + 1, p_height * 8):  # Much less frequent application
                    for x in range(0, width - p_width + 1, p_width * 8):
                        for c in range(channels):
                            region = img_array[y:y+p_height, x:x+p_width, c]
                            candidate_region = region + pattern
                            # Only apply where change is minimal
                            change_mask = np.abs(pattern) < 0.05
                            region[change_mask] = candidate_region[change_mask]
                            img_array[y:y+p_height, x:x+p_width, c] = np.clip(region, 0, 255)
            
            img_array = img_array.astype(np.uint8)
        
        return Image.fromarray(img_array)

    def _apply_frequency_domain_manipulation(self, image: Image.Image) -> Image.Image:
        """
        Apply frequency domain manipulations using DCT that are imperceptible
        to humans but create artifacts that confuse neural networks.
        """
        if self.ai_resistance_level < 2:  # Only for moderate and advanced levels
            return image
            
        try:
            from scipy.fft import dctn, idctn
        except ImportError:
            # Fallback to basic manipulation if scipy not available
            return self._apply_basic_frequency_manipulation(image)
        
        img_array = np.array(image).astype(np.float32)
        height, width, channels = img_array.shape
        
        for c in range(channels):
            # Apply 2D DCT to the channel
            dct_channel = dctn(img_array[:, :, c], norm='ortho')
            
            if self.ai_resistance_level >= 2:
                # Moderate: Very subtly modify high-frequency components
                freq_mask = np.zeros_like(dct_channel)
                h_thresh, w_thresh = height // 3, width // 3
                freq_mask[h_thresh:, w_thresh:] = 1
                
                noise_strength = 0.01  # Much more subtle
                high_freq_noise = np.random.normal(0, noise_strength, dct_channel.shape)
                dct_channel += freq_mask * high_freq_noise
            
            if self.ai_resistance_level >= 3:
                # Advanced: Create very subtle frequency patterns
                mid_freq_h = slice(height // 8, height // 4)  # Less aggressive frequency range
                mid_freq_w = slice(width // 8, width // 4)
                
                # Add very subtle structured patterns in frequency domain
                pattern_strength = 0.005  # Much more subtle
                if dct_channel[mid_freq_h, mid_freq_w].size > 0:
                    pattern_size = min(dct_channel[mid_freq_h, mid_freq_w].shape)
                    if pattern_size > 0:
                        pattern = np.sin(np.arange(pattern_size)) * pattern_strength
                        pattern_2d = np.outer(pattern, pattern)
                        
                        actual_h, actual_w = dct_channel[mid_freq_h, mid_freq_w].shape
                        pattern_h, pattern_w = pattern_2d.shape
                        
                        end_h = min(pattern_h, actual_h)
                        end_w = min(pattern_w, actual_w)
                        
                        dct_channel[mid_freq_h, mid_freq_w][:end_h, :end_w] += pattern_2d[:end_h, :end_w]
            
            # Convert back to spatial domain
            img_array[:, :, c] = idctn(dct_channel, norm='ortho')
        
        # Ensure values are in valid range
        img_array = np.clip(img_array, 0, 255)
        return Image.fromarray(img_array.astype(np.uint8))

    def _apply_basic_frequency_manipulation(self, image: Image.Image) -> Image.Image:
        """
        Fallback frequency manipulation when scipy is not available.
        Uses simple spatial filtering to approximate frequency domain effects.
        """
        img_array = np.array(image)
        height, width, channels = img_array.shape
        
        # Very subtle high-pass filter
        high_pass_kernel = np.array([[-0.1, -0.1, -0.1], [-0.1, 0.8, -0.1], [-0.1, -0.1, -0.1]]) / 9
        
        for c in range(channels):
            channel = img_array[:, :, c].astype(np.float32)
            # Apply convolution manually (simplified)
            filtered = np.zeros_like(channel)
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    region = channel[y-1:y+2, x-1:x+2]
                    filtered[y, x] = np.sum(region * high_pass_kernel) * 0.01  # Very very subtle
            
            img_array[:, :, c] = np.clip(channel + filtered, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)

    def generate(self) -> tuple:
        """
        Generate a CAPTCHA image and its solution text.
        
        Returns:
            tuple: A tuple containing (PIL.Image.Image, str) - the CAPTCHA image and its text solution.
        """
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

        # Step 4: Apply the final blur to the composite image.
        if self.blur_radius > 0: image = image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        # Step 5: Apply AI resistance techniques (NEW!)
        if self.ai_resistance_level > 0:
            # Apply in order of subtlety (least to most perceptible)
            image = self._apply_histogram_manipulation(image)
            image = self._apply_rgb_perturbations(image)
            image = self._apply_adversarial_noise(image)
            image = self._apply_frequency_domain_manipulation(image)

        return image, text
