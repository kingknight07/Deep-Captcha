# DeepCaptcha

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)

DeepCaptcha is a Python library for generating CAPTCHA images. It includes features to add noise and distortions that provide resistance against automated recognition (OCR and machine learning models) while remaining readable by humans.

## Features

- **Adjustable Resistance**: Supports different levels of noise and perturbations.
- **Customizable**: Control image dimensions, text length, fonts, and more.
- **Color Support**: Generate both color and grayscale CAPTCHAs.
- **Resource Efficient**: Designed for low-latency generation.

## Installation

```bash
pip install deepcaptcha
```

## Quick Start

```python
from deepcaptcha import DeepCaptcha

# Create a generator instance
gen = DeepCaptcha(ai_resistance_level=1)

# Generate an image and its corresponding text
image, text = gen.generate()

# Save the result
image.save("captcha.png")
print(f"CAPTCHA text: {text}")
```

## Configuration

You can customize the generation process using various parameters:

```python
captcha = DeepCaptcha(
    width=300,
    height=100,
    text_length=5,
    color_mode=True,
    ai_resistance_level=2,
    num_lines=2,
    noise_density=0.5
)
```

## Dataset & Research

For those interested in the underlying research and datasets used to test this library:

*   **Hugging Face**: [Deep_Captcha Dataset](https://huggingface.co/datasets/Knight07/Deep_Captcha)
*   **IEEE DataPort**: [Deep-Captcha Data](https://ieee-dataport.org/documents/deep-captcha)

## License

This project is licensed under the MIT License.

## Citation

```bibtex
@software{deepcaptcha2024,
  title={DeepCaptcha: A Python CAPTCHA Library with AI Resistance},
  author={Ayush Shukla},
  year={2024},
  url={https://github.com/kingknight07/Deep-Captcha}
}
```
