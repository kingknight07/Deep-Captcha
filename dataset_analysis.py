#!/usr/bin/env python3
"""
DeepCaptcha Dataset Analysis and ML Testing Helper
Provides utilities for analyzing the generated dataset and preparing it for ML model testing.
"""

import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Tuple
import random


def load_dataset_metadata(metadata_file: str) -> List[Dict]:
    """Load dataset metadata from JSON file."""
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_dataset_distribution(metadata: List[Dict]) -> Dict:
    """Analyze the distribution of dataset characteristics."""
    analysis = {
        'ai_resistance_levels': Counter(),
        'text_lengths': Counter(),
        'color_modes': Counter(),
        'image_dimensions': Counter(),
        'characters_used': Counter()
    }
    
    for item in metadata:
        # AI resistance levels
        analysis['ai_resistance_levels'][item['ai_resistance_level']] += 1
        
        # Text lengths
        analysis['text_lengths'][len(item['text'])] += 1
        
        # Color modes
        color_mode = 'color' if item['parameters']['color_mode'] else 'bw'
        analysis['color_modes'][color_mode] += 1
        
        # Image dimensions
        dims = f"{item['parameters']['width']}x{item['parameters']['height']}"
        analysis['image_dimensions'][dims] += 1
        
        # Characters used
        for char in item['text']:
            analysis['characters_used'][char] += 1
    
    return analysis


def create_train_test_split(
    metadata: List[Dict], 
    train_ratio: float = 0.8,
    stratify_by: str = 'ai_resistance_level'
) -> Tuple[List[Dict], List[Dict]]:
    """Create stratified train-test split."""
    
    if stratify_by is None:
        # Simple random split
        random.shuffle(metadata)
        split_idx = int(len(metadata) * train_ratio)
        return metadata[:split_idx], metadata[split_idx:]
    
    # Group by stratification key
    groups = {}
    for item in metadata:
        key = item[stratify_by]
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    
    train_data = []
    test_data = []
    
    # Split each group proportionally
    for group_items in groups.values():
        random.shuffle(group_items)
        split_idx = int(len(group_items) * train_ratio)
        train_data.extend(group_items[:split_idx])
        test_data.extend(group_items[split_idx:])
    
    return train_data, test_data


def generate_ml_ready_splits(dataset_dir: str, output_dir: str = "ml_splits"):
    """Generate ML-ready train/test splits."""
    print("🔄 Generating ML-ready dataset splits...")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load complete metadata
    metadata_file = os.path.join(dataset_dir, 'metadata', 'complete_dataset_metadata.json')
    metadata = load_dataset_metadata(metadata_file)
    
    # Create different split strategies
    splits = {
        'ai_resistance_stratified': create_train_test_split(metadata, stratify_by='ai_resistance_level'),
        'text_length_stratified': create_train_test_split(metadata, stratify_by='text'),
        'random_split': create_train_test_split(metadata, stratify_by=None)
    }
    
    for split_name, (train_data, test_data) in splits.items():
        # Save train metadata
        train_file = os.path.join(output_dir, f'{split_name}_train.json')
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2)
        
        # Save test metadata
        test_file = os.path.join(output_dir, f'{split_name}_test.json')
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"   📊 {split_name}: {len(train_data)} train, {len(test_data)} test")
    
    print(f"💾 ML splits saved to: {output_dir}")


def create_dataset_visualization(metadata: List[Dict], output_file: str = "dataset_analysis.png"):
    """Create comprehensive dataset visualization."""
    print("📊 Creating dataset visualization...")
    
    analysis = analyze_dataset_distribution(metadata)
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('DeepCaptcha Dataset Analysis', fontsize=16, fontweight='bold')
    
    # AI Resistance Levels
    levels = list(analysis['ai_resistance_levels'].keys())
    counts = list(analysis['ai_resistance_levels'].values())
    axes[0, 0].bar([f'Level {l}' for l in levels], counts, 
                   color=['red', 'orange', 'yellow', 'green'])
    axes[0, 0].set_title('AI Resistance Levels')
    axes[0, 0].set_ylabel('Count')
    
    # Text Lengths
    lengths = sorted(analysis['text_lengths'].keys())
    length_counts = [analysis['text_lengths'][l] for l in lengths]
    axes[0, 1].bar([f'{l} chars' for l in lengths], length_counts, color='skyblue')
    axes[0, 1].set_title('Text Length Distribution')
    axes[0, 1].set_ylabel('Count')
    
    # Color Modes
    modes = list(analysis['color_modes'].keys())
    mode_counts = list(analysis['color_modes'].values())
    axes[0, 2].pie(mode_counts, labels=[m.title() for m in modes], autopct='%1.1f%%',
                   colors=['lightcoral', 'lightgray'])
    axes[0, 2].set_title('Color Mode Distribution')
    
    # Image Dimensions (top 10)
    top_dims = dict(analysis['image_dimensions'].most_common(10))
    axes[1, 0].barh(list(top_dims.keys()), list(top_dims.values()), color='lightgreen')
    axes[1, 0].set_title('Top 10 Image Dimensions')
    axes[1, 0].set_xlabel('Count')
    
    # Character Frequency (top 15)
    top_chars = dict(analysis['characters_used'].most_common(15))
    axes[1, 1].bar(list(top_chars.keys()), list(top_chars.values()), color='gold')
    axes[1, 1].set_title('Top 15 Character Frequency')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # AI Resistance vs Text Length Heatmap
    ai_text_matrix = np.zeros((4, 8))  # 4 AI levels, up to 8 char lengths
    for item in metadata:
        ai_level = item['ai_resistance_level']
        text_len = min(len(item['text']), 7)  # Cap at 7 for visualization
        ai_text_matrix[ai_level, text_len] += 1
    
    im = axes[1, 2].imshow(ai_text_matrix, cmap='Blues', aspect='auto')
    axes[1, 2].set_title('AI Resistance vs Text Length')
    axes[1, 2].set_xlabel('Text Length')
    axes[1, 2].set_ylabel('AI Resistance Level')
    axes[1, 2].set_xticks(range(8))
    axes[1, 2].set_yticks(range(4))
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Dataset visualization saved to: {output_file}")
    plt.close()


def generate_ml_usage_guide(dataset_dir: str):
    """Generate a comprehensive ML usage guide."""
    guide = """
# DeepCaptcha Dataset - ML Usage Guide

## 📁 Dataset Structure
```
deepcaptcha_dataset/
├── level_0_baseline/     # 600 images - No AI resistance
├── level_1_basic/        # 600 images - Basic AI resistance (PSNR 44.8dB)
├── level_2_moderate/     # 600 images - Moderate AI resistance (PSNR 39.1dB)
├── level_3_advanced/     # 600 images - Advanced AI resistance (PSNR 40.4dB)
├── mixed_levels/         # 600 images - Mixed resistance levels
└── metadata/             # JSON metadata files
```

## 🧠 ML Model Testing Strategies

### 1. Baseline Evaluation
- Train on `level_0_baseline` images only
- Test on all levels to measure AI resistance effectiveness
- Expected: High accuracy on Level 0, degraded performance on Levels 1-3

### 2. Robustness Training
- Train on `mixed_levels` for general robustness
- Test on individual levels to measure specific resistance
- Expected: Better overall robustness but potentially lower peak accuracy

### 3. Progressive Difficulty Testing
- Train on Level 0, test on Levels 1, 2, 3 progressively
- Measure performance degradation as AI resistance increases
- Expected: Progressive accuracy decline with resistance level

### 4. Transfer Learning Evaluation
- Pre-train on Level 0, fine-tune on higher levels
- Test adaptation capability of existing models
- Expected: Improved performance with adaptation

## 📊 Evaluation Metrics

### Primary Metrics
- **Character Accuracy**: Per-character recognition accuracy
- **Sequence Accuracy**: Complete CAPTCHA sequence accuracy
- **Resistance Effectiveness**: Accuracy drop from Level 0 to Level 3

### Secondary Metrics
- **Processing Time**: Model inference speed per image
- **Memory Usage**: Model resource requirements
- **Generalization**: Performance on unseen parameter combinations

## 💡 Suggested Experiments

### Experiment 1: CNN Baseline
```python
# Test standard CNN architectures
models = ['ResNet', 'VGG', 'EfficientNet']
for model in models:
    train_on_level_0()
    test_on_all_levels()
    measure_resistance_effectiveness()
```

### Experiment 2: Data Augmentation Impact
```python
# Compare with/without augmentation
augmentation_strategies = ['none', 'standard', 'adversarial']
for strategy in augmentation_strategies:
    train_with_augmentation(strategy)
    evaluate_robustness()
```

### Experiment 3: Multi-Level Training
```python
# Progressive training strategy
for target_level in [0, 1, 2, 3]:
    train_on_level(target_level)
    cross_evaluate_all_levels()
    analyze_specialization_vs_generalization()
```

## 🛡️ AI Resistance Analysis

### Expected Results
- **Level 0**: Vulnerable to all ML attacks
- **Level 1**: 10-20% accuracy drop vs baseline
- **Level 2**: 30-50% accuracy drop vs baseline  
- **Level 3**: 50-70% accuracy drop vs baseline

### Key Insights to Validate
1. **Imperceptibility**: Humans should achieve 95%+ accuracy on all levels
2. **Resistance Gradient**: Performance should degrade with resistance level
3. **Attack Specificity**: Different AI architectures may show varied susceptibility
4. **Adaptation Potential**: Models may learn to overcome lower resistance levels

## 📋 Implementation Template

```python
import json
from PIL import Image
import torch
import torchvision.transforms as transforms

def load_dataset_split(split_file):
    with open(split_file, 'r') as f:
        metadata = json.load(f)
    
    images, labels = [], []
    for item in metadata:
        img_path = os.path.join('deepcaptcha_dataset', 
                               get_category_dir(item['ai_resistance_level']),
                               item['filename'])
        img = Image.open(img_path)
        images.append(img)
        labels.append(item['text'])
    
    return images, labels

def evaluate_ai_resistance(model, dataset_dir):
    results = {}
    for level in [0, 1, 2, 3]:
        level_accuracy = test_on_level(model, level)
        results[f'level_{level}'] = level_accuracy
    
    resistance_effectiveness = (results['level_0'] - results['level_3']) / results['level_0']
    results['resistance_effectiveness'] = resistance_effectiveness
    
    return results
```

## 🎯 Success Criteria

A successful AI resistance validation should demonstrate:
1. **Maintained Human Readability**: >95% human accuracy across all levels
2. **Effective AI Confusion**: >50% accuracy drop from Level 0 to Level 3
3. **Graduated Resistance**: Progressive performance degradation
4. **Practical Applicability**: <100ms additional processing time per image

## 📈 Reporting Template

For each experiment, report:
- Model architecture and parameters
- Training data composition and size
- Accuracy metrics per AI resistance level
- Processing time and resource usage
- Resistance effectiveness score
- Recommendations for production deployment
"""
    
    guide_file = os.path.join(dataset_dir, 'ML_USAGE_GUIDE.md')
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide)
    
    print(f"📋 ML usage guide saved to: {guide_file}")


def main():
    """Main analysis function."""
    print("📊 DeepCaptcha Dataset Analysis")
    print("=" * 50)
    
    dataset_dir = "deepcaptcha_dataset"
    
    if not os.path.exists(dataset_dir):
        print("❌ Dataset directory not found. Please run generate_dataset.py first.")
        return
    
    try:
        # Load metadata
        metadata_file = os.path.join(dataset_dir, 'metadata', 'complete_dataset_metadata.json')
        metadata = load_dataset_metadata(metadata_file)
        print(f"📂 Loaded metadata for {len(metadata)} images")
        
        # Create dataset visualization
        create_dataset_visualization(metadata)
        
        # Generate ML-ready splits
        generate_ml_ready_splits(dataset_dir)
        
        # Generate usage guide
        generate_ml_usage_guide(dataset_dir)
        
        # Print summary
        analysis = analyze_dataset_distribution(metadata)
        print("\n📊 Dataset Summary:")
        print(f"   🛡️ AI Resistance Distribution:")
        for level, count in sorted(analysis['ai_resistance_levels'].items()):
            percentage = count / len(metadata) * 100
            print(f"      Level {level}: {count:,} images ({percentage:.1f}%)")
        
        print(f"\n✅ Analysis complete!")
        print(f"📋 Check ML_USAGE_GUIDE.md for comprehensive testing instructions")
        print(f"📊 View dataset_analysis.png for visual analysis")
        print(f"🔄 Use ml_splits/ directory for train/test splits")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()