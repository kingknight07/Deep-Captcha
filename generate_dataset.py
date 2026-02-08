#!/usr/bin/env python3
"""
DeepCaptcha Dataset Generator
Generates 3000 CAPTCHA images for ML model testing and evaluation.
"""

import os
import sys
import json
import time
from typing import Dict, List
import random
import numpy as np
from PIL import Image

# Add the current directory to path for importing DeepCaptcha
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from DeepCaptcha import DeepCaptcha


def create_dataset_directory(base_dir: str = "deepcaptcha_dataset") -> str:
    """Create the dataset directory structure."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"📁 Created dataset directory: {base_dir}")
    
    # Create subdirectories for different resistance levels
    subdirs = [
        "level_0_baseline",
        "level_1_basic", 
        "level_2_moderate",
        "level_3_advanced",
        "mixed_levels",
        "metadata"
    ]
    
    for subdir in subdirs:
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
            print(f"📁 Created subdirectory: {subdir}")
    
    return base_dir


def generate_captcha_batch(
    count: int,
    ai_resistance_level: int,
    output_dir: str,
    prefix: str,
    vary_parameters: bool = True
) -> List[Dict]:
    """Generate a batch of CAPTCHA images with metadata."""
    
    print(f"🔄 Generating {count} CAPTCHAs (Level {ai_resistance_level})...")
    
    metadata_list = []
    start_time = time.time()
    
    for i in range(count):
        # Vary parameters to create diverse dataset
        if vary_parameters:
            config = {
                'width': random.choice([250, 280, 300]),
                'height': random.choice([80, 100, 120]),
                'text_length': random.choice([4, 5, 6]),
                'num_lines': random.randint(3, 8),
                'line_thickness': random.randint(1, 4),
                'blur_level': random.uniform(0.1, 0.8),
                'noise_density': random.uniform(0.2, 0.6),
                'color_mode': random.choice([True, False]),
                'ai_resistance_level': ai_resistance_level
            }
        else:
            # Standard configuration
            config = {
                'width': 280,
                'height': 100, 
                'text_length': 5,
                'num_lines': 5,
                'line_thickness': 2,
                'blur_level': 0.3,
                'noise_density': 0.3,
                'color_mode': True,
                'ai_resistance_level': ai_resistance_level
            }
        
        # Create CAPTCHA instance
        captcha = DeepCaptcha(**config)
        
        # Generate CAPTCHA
        img, text = captcha.generate()
        
        # Save image
        filename = f"{prefix}_{i:04d}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        
        # Store metadata
        metadata = {
            'filename': filename,
            'text': text,
            'ai_resistance_level': ai_resistance_level,
            'parameters': config,
            'timestamp': time.time()
        }
        metadata_list.append(metadata)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = (count - i - 1) * avg_time
            print(f"   📊 Progress: {i+1}/{count} ({(i+1)/count*100:.1f}%) "
                  f"- ETA: {remaining:.1f}s")
    
    elapsed_total = time.time() - start_time
    print(f"   ✅ Completed {count} images in {elapsed_total:.1f}s "
          f"(avg: {elapsed_total/count*1000:.1f}ms per image)")
    
    return metadata_list


def generate_mixed_dataset(count: int, output_dir: str) -> List[Dict]:
    """Generate a mixed dataset with random AI resistance levels."""
    print(f"🔄 Generating {count} mixed-level CAPTCHAs...")
    
    metadata_list = []
    start_time = time.time()
    
    for i in range(count):
        # Randomly select AI resistance level
        ai_level = random.choice([0, 1, 2, 3])
        
        # Vary all parameters for maximum diversity
        config = {
            'width': random.choice([200, 250, 280, 300, 350]),
            'height': random.choice([60, 80, 100, 120, 140]),
            'text_length': random.choice([3, 4, 5, 6, 7]),
            'num_lines': random.randint(2, 10),
            'line_thickness': random.randint(1, 5),
            'blur_level': random.uniform(0.0, 1.0),
            'noise_density': random.uniform(0.0, 0.8),
            'color_mode': random.choice([True, False]),
            'ai_resistance_level': ai_level
        }
        
        # Create CAPTCHA instance
        captcha = DeepCaptcha(**config)
        
        # Generate CAPTCHA
        img, text = captcha.generate()
        
        # Save image
        filename = f"mixed_{i:04d}_level{ai_level}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        
        # Store metadata
        metadata = {
            'filename': filename,
            'text': text,
            'ai_resistance_level': ai_level,
            'parameters': config,
            'timestamp': time.time()
        }
        metadata_list.append(metadata)
        
        # Progress indicator
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = (count - i - 1) * avg_time
            print(f"   📊 Progress: {i+1}/{count} ({(i+1)/count*100:.1f}%) "
                  f"- ETA: {remaining:.1f}s")
    
    elapsed_total = time.time() - start_time
    print(f"   ✅ Completed {count} mixed images in {elapsed_total:.1f}s "
          f"(avg: {elapsed_total/count*1000:.1f}ms per image)")
    
    return metadata_list


def save_metadata(metadata_list: List[Dict], output_file: str):
    """Save metadata to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved metadata to: {output_file}")


def generate_dataset_summary(base_dir: str, all_metadata: Dict[str, List[Dict]]):
    """Generate a comprehensive dataset summary."""
    summary = {
        'dataset_info': {
            'name': 'DeepCaptcha ML Testing Dataset',
            'version': '1.0',
            'total_images': sum(len(meta) for meta in all_metadata.values()),
            'generation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'library_version': '11.0 - AI Resistance Edition'
        },
        'categories': {},
        'statistics': {
            'ai_resistance_levels': {str(i): 0 for i in range(4)},
            'color_modes': {'color': 0, 'bw': 0},
            'text_lengths': {},
            'image_dimensions': {}
        }
    }
    
    # Analyze each category
    for category, metadata_list in all_metadata.items():
        summary['categories'][category] = {
            'count': len(metadata_list),
            'description': get_category_description(category)
        }
        
        # Collect statistics
        for item in metadata_list:
            # AI resistance levels
            level = str(item['ai_resistance_level'])
            summary['statistics']['ai_resistance_levels'][level] += 1
            
            # Color modes
            if item['parameters']['color_mode']:
                summary['statistics']['color_modes']['color'] += 1
            else:
                summary['statistics']['color_modes']['bw'] += 1
            
            # Text lengths
            text_len = str(len(item['text']))
            summary['statistics']['text_lengths'][text_len] = \
                summary['statistics']['text_lengths'].get(text_len, 0) + 1
            
            # Image dimensions
            dims = f"{item['parameters']['width']}x{item['parameters']['height']}"
            summary['statistics']['image_dimensions'][dims] = \
                summary['statistics']['image_dimensions'].get(dims, 0) + 1
    
    # Save summary
    summary_file = os.path.join(base_dir, 'metadata', 'dataset_summary.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📋 Saved dataset summary to: {summary_file}")
    
    return summary


def get_category_description(category: str) -> str:
    """Get description for each category."""
    descriptions = {
        'level_0_baseline': 'No AI resistance - vulnerable to ML attacks',
        'level_1_basic': 'Basic AI resistance - imperceptible protection (PSNR 44.8dB)',
        'level_2_moderate': 'Moderate AI resistance - enhanced security (PSNR 39.1dB)',
        'level_3_advanced': 'Advanced AI resistance - maximum protection (PSNR 40.4dB)',
        'mixed_levels': 'Mixed AI resistance levels with varied parameters'
    }
    return descriptions.get(category, 'Unknown category')


def print_dataset_statistics(summary: Dict):
    """Print comprehensive dataset statistics."""
    print("\n" + "=" * 60)
    print("📊 DATASET GENERATION COMPLETE")
    print("=" * 60)
    
    info = summary['dataset_info']
    print(f"📁 Dataset: {info['name']}")
    print(f"🔢 Total Images: {info['total_images']:,}")
    print(f"📅 Generated: {info['generation_date']}")
    print(f"⚡ Library Version: {info['library_version']}")
    
    print(f"\n📂 Categories:")
    for category, data in summary['categories'].items():
        print(f"   {category}: {data['count']:,} images - {data['description']}")
    
    print(f"\n🛡️ AI Resistance Distribution:")
    for level, count in summary['statistics']['ai_resistance_levels'].items():
        percentage = count / info['total_images'] * 100
        print(f"   Level {level}: {count:,} images ({percentage:.1f}%)")
    
    print(f"\n🎨 Color Mode Distribution:")
    color_stats = summary['statistics']['color_modes']
    total = sum(color_stats.values())
    for mode, count in color_stats.items():
        percentage = count / total * 100
        print(f"   {mode.title()}: {count:,} images ({percentage:.1f}%)")
    
    print(f"\n📝 Text Length Distribution:")
    for length, count in sorted(summary['statistics']['text_lengths'].items()):
        percentage = count / info['total_images'] * 100
        print(f"   {length} chars: {count:,} images ({percentage:.1f}%)")
    
    print(f"\n🖼️ Top Image Dimensions:")
    dim_stats = sorted(summary['statistics']['image_dimensions'].items(), 
                      key=lambda x: x[1], reverse=True)[:5]
    for dims, count in dim_stats:
        percentage = count / info['total_images'] * 100
        print(f"   {dims}: {count:,} images ({percentage:.1f}%)")


def main():
    """Main dataset generation function."""
    print("🚀 DeepCaptcha Dataset Generator")
    print("=" * 60)
    print("Generating 3000 CAPTCHA images for ML model testing...")
    print()
    
    # Create dataset directory
    base_dir = create_dataset_directory()
    
    # Define dataset distribution
    dataset_config = [
        # (count, ai_level, subdir, prefix, vary_params)
        (600, 0, "level_0_baseline", "baseline", True),
        (600, 1, "level_1_basic", "basic", True), 
        (600, 2, "level_2_moderate", "moderate", True),
        (600, 3, "level_3_advanced", "advanced", True),
        # Mixed dataset with remaining 600 images
    ]
    
    all_metadata = {}
    total_start_time = time.time()
    
    try:
        # Generate datasets for each AI resistance level
        for count, ai_level, subdir, prefix, vary_params in dataset_config:
            subdir_path = os.path.join(base_dir, subdir)
            metadata = generate_captcha_batch(
                count=count,
                ai_resistance_level=ai_level,
                output_dir=subdir_path,
                prefix=prefix,
                vary_parameters=vary_params
            )
            all_metadata[subdir] = metadata
            
            # Save individual metadata
            metadata_file = os.path.join(base_dir, 'metadata', f'{subdir}_metadata.json')
            save_metadata(metadata, metadata_file)
            print()
        
        # Generate mixed dataset
        mixed_dir = os.path.join(base_dir, "mixed_levels")
        mixed_metadata = generate_mixed_dataset(600, mixed_dir)
        all_metadata["mixed_levels"] = mixed_metadata
        
        # Save mixed metadata
        mixed_metadata_file = os.path.join(base_dir, 'metadata', 'mixed_levels_metadata.json')
        save_metadata(mixed_metadata, mixed_metadata_file)
        
        # Generate and save comprehensive metadata
        complete_metadata = []
        for metadata_list in all_metadata.values():
            complete_metadata.extend(metadata_list)
        
        complete_metadata_file = os.path.join(base_dir, 'metadata', 'complete_dataset_metadata.json')
        save_metadata(complete_metadata, complete_metadata_file)
        
        # Generate dataset summary
        summary = generate_dataset_summary(base_dir, all_metadata)
        
        # Print statistics
        print_dataset_statistics(summary)
        
        total_time = time.time() - total_start_time
        print(f"\n⏱️ Total Generation Time: {total_time:.1f}s")
        print(f"📈 Average Speed: {len(complete_metadata)/total_time:.1f} images/second")
        
        print(f"\n🎯 Usage Instructions:")
        print(f"   📁 Dataset Location: {os.path.abspath(base_dir)}")
        print(f"   📋 Metadata Files: {os.path.join(base_dir, 'metadata')}")
        print(f"   🔬 Test ML Models: Use different AI resistance levels to evaluate robustness")
        print(f"   📊 Benchmark Performance: Compare accuracy across resistance levels")
        
        print(f"\n✅ Dataset Generation Successful!")
        print(f"🛡️ Ready for AI resistance testing and ML model evaluation!")
        
    except Exception as e:
        print(f"❌ Error during dataset generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()