# DeepCaptcha: An AI-Resistant CAPTCHA Generation Framework

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![PyPI](https://img.shields.io/pypi/v/deepcaptcha.svg)
![Security](https://img.shields.io/badge/Security-AI--Resistant-blue.svg)

## Overview
DeepCaptcha is a Python-based framework designed for the generation of adversarial CAPTCHAs. Unlike traditional CAPTCHA systems which are increasingly vulnerable to Deep Learning-based recognition attacks, DeepCaptcha implements multi-level AI resistance techniques while maintaining high human readability.

## 📊 Dataset & Benchmarks
The research dataset associated with this project is available on the following platforms:

*   **Hugging Face**: [Deep_Captcha Dataset](https://huggingface.co/datasets/Knight07/Deep_Captcha)
*   **IEEE DataPort**: [Deep-Captcha Data](https://ieee-dataport.org/documents/deep-captcha)

### Dataset Analysis
The framework includes a diverse dataset of 3,000 images, categorized by AI resistance levels (Baseline to Advanced).

![Dataset Distribution](dataset_analysis.png)

## 🔬 Research & Evaluation Results
The effectiveness of DeepCaptcha has been validated through extensive benchmarking against modern Convolutional Neural Network (CNN) architectures and standard OCR systems.

### Accuracy Heatmap
Detailed analysis of recognition performance across varying character lengths and AI resistance levels:
![Accuracy Heatmap](research_results/advanced_figures/viz_accuracy_heatmap_detailed.png)

### Performance Distribution
Character-wise accuracy distribution demonstrating the graduated resistance levels:
![Character Accuracy Distribution](research_results/advanced_figures/viz_char_accuracy_distribution.png)

### Reliability Analysis
Confidence vs. Reliability metrics for recognition models attempting to solve DeepCaptcha outputs:
![Confidence Reliability](research_results/advanced_figures/viz_confidence_reliability.png)

## 🛡️ AI Resistance Methodology
DeepCaptcha employs a progressive defense strategy to protect against automated recognition:

*   **Spatial Perturbations**: Strategic shear and distortion to disrupt CNN spatial recognition.
*   **Frequency Domain Defense**: Implementation of DCT-based perturbations (Imperceptible to humans, PSNR > 40dB).
*   **Adversarial Noise**: Histogram-based transformations and targeted pixel perturbations.

### Resistance Tiers
| Level | Methodology | Resistance Index | Human Accuracy |
|-------|-------------|------------------|----------------|
| **0** | Baseline | Low | >99% |
| **1** | Basic | Moderate | >98% |
| **2** | Moderate | High | >96% |
| **3** | Advanced | Maximum | >95% |

## 🚀 Technical Implementation

### Installation
```bash
pip install deepcaptcha
```

### Basic Integration
```python
from deepcaptcha import DeepCaptcha

# Initialize with Advanced AI resistance
captcha_gen = DeepCaptcha(ai_resistance_level=3)

# Generate CAPTCHA image and solution string
image, solution = captcha_gen.generate()
image.save("protected_captcha.png")
```

### Configuration Parameters
The framework supports comprehensive customization for various security and accessibility requirements:
```python
captcha = DeepCaptcha(
    width=350,              # Image dimensions
    height=120,
    text_length=5,          # Sequence length
    color_mode=True,        # Color vs Grayscale optimization
    ai_resistance_level=3,  # 0 to 3
    num_lines=3,            # Strategic strike lines
    noise_density=0.6       # Background perturbations
)
```

## 📈 Performance Characteristics
DeepCaptcha is optimized for production environments, ensuring low latency even with maximum protection:
- **Mean Generation Latency**: ~7.8ms (Level 3)
- **Memory Footprint**: < 60KB per instance
- **Human Compatibility**: Validated human imperceptibility (PSNR > 40dB)

## 📁 Project Repository Structure
- `deepcaptcha/`: Core library implementation and resources.
- `research_results/`: Detailed benchmark figures and analysis reports.
- `benchmark_comparison.py`: Scripts for replicating experimental results.

## 📄 License
This intellectual property is licensed under the **MIT License**.

## 📚 Citation
If you utilize this framework or the associated dataset in an academic or industrial context, please cite as follows:

```bibtex
@software{deepcaptcha2024,
  title={DeepCaptcha: An AI-Resistant CAPTCHA Generation Framework},
  author={Ayush Shukla},
  year={2024},
  url={https://github.com/kingknight07/Deep-Captcha}
}
```
