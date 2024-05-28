import os
import cv2
from google.cloud import translate_v2 as translate
from PIL import Image, ImageDraw, ImageFont

def translate_text(text, target_language='vi'):
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

def write_text_on_image(image_path, text, coordinates, output_path):
    # Open the image with PIL
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Choose a font and size
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font_size = 20
    font = ImageFont.truetype(font_path, font_size)
    
    # Write each piece of text on the image at the specified coordinates
    for (t, (x, y)) in zip(text, coordinates):
        draw.text((x, y), t, fill="black", font=font)
    
    # Save the edited image
    image.save(output_path)

def process_and_translate(img_dir, ocr_dir, mask_dir, cleaned_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a list of image paths to process
    image_type = ('.png', '.jpg', '.jpeg', '.bmp', '.jpe', '.PNG', '.JPG', '.JPEG', '.JPE', '.BMP')
    image_paths = [os.path.join(img_dir, image) for image in os.listdir(img_dir) if image.endswith(image_type)]
    
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        
        # Assume OCR text and coordinates are stored in a corresponding JSON file
        ocr_file = os.path.join(ocr_dir, os.path.splitext(image_name)[0] + '.json')
        if not os.path.exists(ocr_file):
            print(f"Skipping {image_path}, OCR data not found.")
            continue
        
        # Read the OCR data (assuming a specific format)
        with open(ocr_file, 'r') as f:
            ocr_data = json.load(f)
        
        text = [item['text'] for item in ocr_data]
        coordinates = [(item['x'], item['y']) for item in ocr_data]
        
        # Translate the text to Vietnamese
        translated_text = [translate_text(t) for t in text]
        
        # Write the translated text on the cleaned image
        cleaned_image_path = os.path.join(cleaned_dir, image_name)
        output_image_path = os.path.join(output_dir, image_name)
        
        write_text_on_image(cleaned_image_path, translated_text, coordinates, output_image_path)
        
        print(f'Processed and translated {image_path}')

if __name__ == "__main__":
    img_dir = '/path/to/images'
    ocr_dir = '/path/to/ocr_data'
    mask_dir = '/path/to/save/masks'
    cleaned_dir = '/path/to/cleaned_images'
    output_dir = '/path/to/output_images'
    
    process_and_translate(img_dir, ocr_dir, mask_dir, cleaned_dir, output_dir)
