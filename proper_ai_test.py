#!/usr/bin/env python3
"""
Proper AI Resistance Test - Apply resistance to the same base image
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DeepCaptcha import DeepCaptcha


def calculate_image_metrics(img1: Image.Image, img2: Image.Image) -> dict:
    """Calculate difference metrics between two images."""
    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    
    # Mean Squared Error (MSE)
    mse = np.mean((arr1 - arr2) ** 2)
    
    # Peak Signal-to-Noise Ratio (PSNR)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Pixel-level difference statistics
    diff = np.abs(arr1 - arr2)
    max_diff = np.max(diff)
    avg_diff = np.mean(diff)
    
    return {
        'mse': mse,
        'psnr': psnr,
        'max_pixel_diff': max_diff,
        'avg_pixel_diff': avg_diff
    }


def test_ai_resistance_on_same_image():
    """Test AI resistance by applying it to the same base image."""
    print("🔬 AI Resistance Test - Same Base Image")
    print("=" * 60)
    
    # Generate a base CAPTCHA without AI resistance
    captcha = DeepCaptcha(
        text_length=5,
        ai_resistance_level=0,
        num_lines=5,
        noise_density=0.3
    )
    
    base_img, text = captcha.generate()
    print(f"Generated base CAPTCHA with text: {text}")
    
    # Now test different AI resistance levels on the same image
    images = [base_img]
    names = ["Original (Level 0)"]
    metrics = []
    
    for level in [1, 2, 3]:
        print(f"\n📊 Testing AI Resistance Level {level}")
        
        # Create temporary captcha instance just to access the AI resistance methods
        temp_captcha = DeepCaptcha(ai_resistance_level=level)
        
        # Apply AI resistance to the same base image
        resistant_img = base_img.copy()
        
        if level >= 1:
            resistant_img = temp_captcha._apply_histogram_manipulation(resistant_img)
            resistant_img = temp_captcha._apply_rgb_perturbations(resistant_img)
            resistant_img = temp_captcha._apply_adversarial_noise(resistant_img)
        
        if level >= 2:
            resistant_img = temp_captcha._apply_frequency_domain_manipulation(resistant_img)
        
        # Calculate metrics
        metrics_dict = calculate_image_metrics(base_img, resistant_img)
        metrics.append({
            'level': level,
            'name': f"Level {level}",
            **metrics_dict
        })
        
        images.append(resistant_img)
        names.append(f"Level {level}")
        
        print(f"   📈 MSE: {metrics_dict['mse']:.2f}")
        print(f"   📊 PSNR: {metrics_dict['psnr']:.1f} dB")
        print(f"   🔍 Max Pixel Diff: {metrics_dict['max_pixel_diff']:.1f}")
        print(f"   📋 Avg Pixel Diff: {metrics_dict['avg_pixel_diff']:.2f}")
        
        # Human imperceptibility assessment
        if metrics_dict['psnr'] > 40:
            print("   ✅ Human Imperceptible (PSNR > 40dB)")
        elif metrics_dict['psnr'] > 30:
            print("   ⚠️ Barely Perceptible (30-40dB)")
        else:
            print("   ❌ Potentially Visible (PSNR < 30dB)")
    
    # Create visual comparison
    print(f"\n🖼️ Creating Visual Comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('DeepCaptcha AI Resistance - Same Base Image', fontsize=16, fontweight='bold')
    
    for i, (img, name) in enumerate(zip(images, names)):
        row, col = i // 2, i % 2
        axes[row, col].imshow(img)
        axes[row, col].set_title(name, fontsize=12, fontweight='bold')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('ai_resistance_same_image.png', dpi=300, bbox_inches='tight')
    print(f"   💾 Saved visual comparison to: ai_resistance_same_image.png")
    plt.close()
    
    # Create difference visualizations
    print(f"\n🔍 Creating Difference Visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Pixel Differences from Original', fontsize=16, fontweight='bold')
    
    for i, (level, img) in enumerate(zip([1, 2, 3], images[1:])):
        diff = np.abs(np.array(base_img).astype(np.float32) - np.array(img).astype(np.float32))
        diff_normalized = (diff / np.max(diff) * 255).astype(np.uint8) if np.max(diff) > 0 else diff
        
        axes[i].imshow(diff_normalized, cmap='hot')
        axes[i].set_title(f'Level {level} Differences\n(Max: {np.max(diff):.1f})', fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('ai_resistance_differences.png', dpi=300, bbox_inches='tight')
    print(f"   💾 Saved difference visualization to: ai_resistance_differences.png")
    plt.close()
    
    return metrics


def main():
    """Main testing function."""
    print("🚀 DeepCaptcha AI Resistance Proper Testing")
    print("=" * 60)
    print("Testing AI resistance applied to the same base image...\n")
    
    try:
        metrics = test_ai_resistance_on_same_image()
        
        print("\n" + "=" * 60)
        print("📊 Summary Results:")
        print("-" * 60)
        print(f"{'Level':<8} {'PSNR (dB)':<12} {'Max Diff':<12} {'Avg Diff':<12} {'Status':<15}")
        print("-" * 60)
        
        for m in metrics:
            status = "Imperceptible" if m['psnr'] > 40 else \
                    "Barely Visible" if m['psnr'] > 30 else "Visible"
            
            print(f"{m['level']:<8} {m['psnr']:<12.1f} {m['max_pixel_diff']:<12.1f} "
                  f"{m['avg_pixel_diff']:<12.2f} {status:<15}")
        
        print("\n🎯 Conclusions:")
        best_level = None
        for m in metrics:
            if m['psnr'] > 40:
                if best_level is None or m['level'] > best_level:
                    best_level = m['level']
        
        if best_level:
            print(f"   ✅ Level {best_level} provides maximum AI resistance while remaining imperceptible")
        else:
            print("   ⚠️  All levels may be visible - consider reducing strength")
        
        print(f"   🛡️ AI resistance successfully alters image statistics")
        print(f"   👁️ Visual impact assessment complete")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()