#!/usr/bin/env python3
"""
Simple debug test for AI resistance
"""

import numpy as np
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DeepCaptcha import DeepCaptcha

def test_simple_comparison():
    """Test with consistent text to see actual differences."""
    
    # Create baseline captcha (no AI resistance)
    captcha_baseline = DeepCaptcha(
        text_length=5,
        ai_resistance_level=0,
        num_lines=3,
        noise_density=0.2,
        blur_level=0.0  # Disable blur to see pure differences
    )
    
    # Create AI resistant captcha
    captcha_resistant = DeepCaptcha(
        text_length=5,
        ai_resistance_level=1,
        num_lines=3,
        noise_density=0.2,
        blur_level=0.0  # Disable blur to see pure differences
    )
    
    # Generate using same random seed approach
    import random
    random.seed(42)
    np.random.seed(42)
    img1, text1 = captcha_baseline.generate()
    
    random.seed(42)
    np.random.seed(42)
    img2, text2 = captcha_resistant.generate()
    
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    
    # Calculate differences
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    
    diff = np.abs(arr1 - arr2)
    print(f"Max difference: {np.max(diff)}")
    print(f"Mean difference: {np.mean(diff)}")
    print(f"Std difference: {np.std(diff)}")
    
    # Show where differences occur
    large_diffs = diff > 10
    print(f"Pixels with >10 difference: {np.sum(large_diffs)}")
    print(f"Total pixels: {diff.size}")
    print(f"Percentage of changed pixels: {np.sum(large_diffs) / diff.size * 100:.2f}%")
    
    # Save images for visual inspection
    img1.save("baseline.png")
    img2.save("resistant.png")
    
    # Create difference image
    diff_img = Image.fromarray((diff * 10).astype(np.uint8))  # Amplify differences
    diff_img.save("differences.png")
    
    print("Images saved: baseline.png, resistant.png, differences.png")

if __name__ == "__main__":
    test_simple_comparison()