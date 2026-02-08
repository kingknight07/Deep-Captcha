"""
DeepCaptcha: AI-Resistant Python CAPTCHA Library

A highly configurable, production-ready CAPTCHA generator with
breakthrough adversarial AI resistance technology.

Example:
    >>> from deepcaptcha import DeepCaptcha
    >>> captcha = DeepCaptcha(ai_resistance_level=2)
    >>> image, text = captcha.generate()
    >>> image.save("captcha.png")
"""

from deepcaptcha.core import DeepCaptcha

__version__ = "1.0.0"
__author__ = "Ayush Shukla"
__email__ = "shuklaayush0704@gmail.com"
__all__ = ["DeepCaptcha"]
