# your_app.py

import os
# Import the class from your library file
from  DeepCaptcha import DeepCaptcha


def generate_captcha_examples():
    """
    A function demonstrating how to call and use the DeepCaptcha library.
    """
    output_dir = "Captcha_Dataset"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(10):

        try:
            custom_captcha_gen = DeepCaptcha(
                width=350,  # A wider image
                height=120,  # A taller image
                text_length=4,  # More characters for increased difficulty
                num_lines=3,  # Number of strike lines across the text
                dot_radius=1,  # Use 2px diameter dots instead of single pixels
                blur_level=0.8,  # A very slight blur
                noise_density=0.6  # High density of background noise dots
            )
            image, text = custom_captcha_gen.generate()
            image.save(os.path.join(output_dir, f"{text}.png"))
            print(f"   - {text}.png")
        except RuntimeError as e:
            print(f"   - ERROR: Could not generate custom CAPTCHA. Reason: {e}")

    print(f"\nAll examples generated successfully!")


if __name__ == "__main__":
    generate_captcha_examples()