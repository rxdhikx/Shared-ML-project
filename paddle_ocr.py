from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
from googletrans import Translator
from langdetect import detect

def perform_ocr(image_path):
    """
    Perform OCR on the input image file.

    Args:
    image_path (str): Path to the image file.

    Returns:
    str: Extracted text from OCR.
    """
    # Setup OCR model with English language
    ocr_model = PaddleOCR(lang='en')

    # Check if the image is in HEIC format
    if image_path.lower().endswith('.heic'):
        # Convert HEIC image to PNG format
        heic_img = Image.open(image_path)
        image_np = np.array(heic_img.convert('RGB'))
    else:
        # Load the image
        image = Image.open(image_path)
        # Convert image to numpy array
        image_np = np.array(image)

    # Perform OCR on the image
    result = ocr_model.ocr(image_np)

    # Extract text from OCR result
    text = ""
    if result is not None:
        for line in result:
            for word in line:
                text += word[1][0] + ' '
    else:
        print("No text detected in image:", image_path)

    # Check if the detected text is in English
    if detect(text) != 'en':
        # Translate text to English
        translator = Translator()
        translated = translator.translate(text, src='auto', dest='en')
        text = translated.text

    return text.strip()


def detect_ingredients(text):
    """
    Detect ingredients from the OCR result.

    Args:
    text (str): OCR result text.

    Returns:
    set: Set of detected ingredients.
    """
    ingredients = ['beans', 'salt', 'butter', 'sugar', 'onion', 'water', 'eggs', 'oliveoil', 'flour', 'milk',
                   'garliccloves', 'pepper', 'brownsugar', 'garlic', 'all-purposeflour', 'bakingpowder', 'egg',
                   'saltandpepper', 'parmesancheese', 'lemonjuice', 'bakingsoda', 'vegetableoil', 'vanilla',
                   'blackpepper', 'cinnamon', 'tomatoes', 'sourcream', 'garlicpowder', 'vanillaextract', 'oil',
                   'honey', 'onions', 'creamcheese', 'garlicclove', 'celery', 'cheddarcheese', 'unsaltedbutter',
                   'soysauce']

    # Convert text to lowercase
    text = text.lower()

    # Split text into words
    text = text.split()

    # Find intersection of detected ingredients and predefined ingredients list
    detected_ingredients = set(ingredients).intersection(text)

    return detected_ingredients


# Example usage:
image_path = "/home/rravindra0463@id.sdsu.edu/ml/datasets/beans1.png"
ocr_text = perform_ocr(image_path)
detected_ingredients = detect_ingredients(ocr_text)
print(detected_ingredients)
