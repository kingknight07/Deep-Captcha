
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from typing import List, Dict

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configuration
RESULTS_FILE = "research_results/detailed_results.json"
OUTPUT_DIR = "research_results/advanced_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set visual style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.5)
colors = ["#FF0B04", "#431C53", "#E05915", "#COD40E"] # DeepCaptcha brand-ish colors or high contrast
custom_palette = sns.color_palette("rocket")

def load_data():
    print(f"📂 Loading results from {RESULTS_FILE}...")
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def create_dataframe(data):
    rows = []
    for item in data:
        for model_res in item['model_results']:
            rows.append({
                'Image': item['image_path'],
                'Level': item['ai_resistance_level'],
                'Ground Truth': item['ground_truth'],
                'Model': model_res['model_name'],
                'Prediction': model_res['predicted_text'],
                'Char Accuracy': model_res['character_accuracy'] * 100,
                'Confidence': model_res['confidence'] if model_res['confidence'] is not None else 0.0,
                'Time (ms)': model_res['inference_time_ms'],
                'Is Correct': model_res['is_correct']
            })
    return pd.DataFrame(rows)

def plot_char_accuracy_distribution(df):
    print("📈 Generating Character Accuracy Violin Plot...")
    plt.figure(figsize=(14, 8))
    
    # Violin plot showing distribution shape
    ax = sns.violinplot(x="Level", y="Char Accuracy", hue="Model", data=df, 
                        palette="viridis", inner="quartile", cut=0)
    
    plt.title("Character Accuracy Distribution by AI Resistance Level", fontsize=20, fontweight='bold', pad=20)
    plt.xlabel("AI Resistance Level", fontsize=16)
    plt.ylabel("Character Accuracy (%)", fontsize=16)
    plt.ylim(0, 100)
    plt.legend(title='OCR Model', loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_char_accuracy_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_vs_accuracy(df):
    print("📈 Generating Confidence vs Accuracy Analysis...")
    # Filter only models that provide confidence (usually > 0)
    df_conf = df[df['Confidence'] > 0]
    
    if df_conf.empty:
        print("   ⚠️ No confidence data available, skipping plot.")
        return

    g = sns.lmplot(x="Confidence", y="Char Accuracy", col="Model", hue="Level", data=df_conf,
                   height=5, aspect=1, palette="plasma", scatter_kws={"alpha": 0.4})
    
    g.fig.suptitle("Model Confidence vs. Actual Character Accuracy", y=1.05, fontsize=20, fontweight='bold')
    plt.savefig(f"{OUTPUT_DIR}/viz_confidence_reliability.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_inference_speed(df):
    print("📈 Generating Inference Speed Comparison...")
    plt.figure(figsize=(12, 6))
    
    # Box plot for speed
    sns.boxplot(x="Model", y="Time (ms)", data=df, palette="magma", showfliers=False) # Hide outliers for cleaner view
    
    plt.title("Inference Latency by Model", fontsize=20, fontweight='bold', pad=20)
    plt.ylabel("Inference Time (ms)", fontsize=16)
    plt.xlabel("OCR Model", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_inference_latency.png", dpi=300)
    plt.close()

def create_visual_failure_grid(data, num_samples=5):
    print("🖼️ Generating Visual Failure Analysis Grid...")
    
    # Select random samples from each level
    levels = [0, 1, 2, 3]
    selected_samples = []
    
    for level in levels:
        level_samples = [d for d in data if d['ai_resistance_level'] == level]
        if level_samples:
            selected_samples.extend(np.random.choice(level_samples, 1)) # 1 per level
    
    if not selected_samples:
        return

    # Create a composite image
    # Layout: Image | Ground Truth | Model 1 Pred | Model 2 Pred | Model 3 Pred
    rows = len(selected_samples)
    cols = 2 # Image + Text Info column
    
    fig = plt.figure(figsize=(15, 4 * rows))
    gs = gridspec.GridSpec(rows, 2, width_ratios=[1, 2])
    
    for idx, sample in enumerate(selected_samples):
        # Load Image
        img_path = sample['image_path']
        try:
            img = Image.open(img_path)
            ax_img = plt.subplot(gs[idx, 0])
            ax_img.imshow(img)
            ax_img.axis('off')
            ax_img.set_title(f"Level {sample['ai_resistance_level']}\nGT: {sample['ground_truth']}", 
                            fontsize=14, color='darkgreen', fontweight='bold')
        except:
            continue
            
        # Text Info
        ax_text = plt.subplot(gs[idx, 1])
        ax_text.axis('off')
        
        # Build text string
        info_text = []
        model_results = sample['model_results']
        
        # Sort by model name to be consistent
        model_results.sort(key=lambda x: x['model_name'])
        
        y_pos = 0.8
        for res in model_results:
            model_name = res['model_name']
            pred = res['predicted_text']
            # Highlight if empty
            if not pred: pred = "[EMPTY]"
            
            # Color code
            color = 'red' if pred != sample['ground_truth'] else 'green'
            
            ax_text.text(0.1, y_pos, f"{model_name}:", fontsize=16, fontweight='bold', color='#333')
            ax_text.text(0.4, y_pos, f"{pred}", fontsize=16, fontweight='bold', color=color, fontfamily='monospace')
            y_pos -= 0.2
            
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_visual_failure_grid.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_heatmap(df):
    print("📈 Generating Model Performance Heatmap...")
    # Pivot for heatmap: Level vs Model
    pivot = df.pivot_table(index="Model", columns="Level", values="Char Accuracy", aggfunc="mean")
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100, 
                linewidths=1, linecolor='white', cbar_kws={'label': 'Char Accuracy (%)'})
    
    plt.title("Mean Character Accuracy Heatmap", fontsize=20, fontweight='bold', pad=20)
    plt.ylabel("OCR Model", fontsize=16)
    plt.xlabel("AI Resistance Level", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/viz_accuracy_heatmap_detailed.png", dpi=300)
    plt.close()

def main():
    try:
        data = load_data()
        df = create_dataframe(data)
        
        print(f"📊 Processed {len(df)} predictions.")
        
        plot_char_accuracy_distribution(df)
        plot_accuracy_heatmap(df)
        plot_inference_speed(df)
        plot_confidence_vs_accuracy(df)
        create_visual_failure_grid(data)
        
        print("\n✅ All advanced figures generated in:")
        print(f"   {os.path.abspath(OUTPUT_DIR)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
