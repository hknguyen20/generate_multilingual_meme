import cv2
import numpy as np
import easyocr
import os
import argparse
from PIL import Image, ImageEnhance, ImageFont, ImageDraw
import json
from googletrans import Translator
from matplotlib import font_manager

################### TEXT REMOVE ##########################

def get_meme_text(image,image_ocr_path):
    text_coordinates = {}
    full = reader.readtext(image, paragraph=True)
    for bbox, text in full:
        xmin = round(min([p[0] for p in bbox])) #round to integer for slicing later
        xmax = round(max([p[0] for p in bbox]))
        ymin = round(min([p[1] for p in bbox]))
        ymax = round(max([p[1] for p in bbox]))
        coord = (xmin, ymin, xmax, ymax)
        text_coordinates[text]=coord
    with open(image_ocr_path, 'w') as f:
        json.dump(text_coordinates, f)
    print(f'Done ocr, written to {image_ocr_path}')
    return text_coordinates.values()

def get_mask_expanded(mask, shift_amount):
    """
    Apply shifts in various directions to make text masks "fatter". 
    This helps prevent inpainting algo from reconstructing the original text 
    rather than inpainting from the surrounding image.

    Parameters:
    mask (numpy.ndarray): A 2D binary numpy array where 1 represents text pixels and 0 represents image pixels.
    shift_amount (int): The number of pixels by which the mask will be shifted in each direction.

    Returns:
    numpy.ndarray: A new binary mask covering more surrounding pixels of the text.
    """

    #horizontal, vertical, diagonal shifts
    shifts = [(offset_x, offset_y) for offset_x in (0, shift_amount, -shift_amount) 
                   for offset_y in (0, shift_amount, -shift_amount)]
    h, w = mask.shape
    for offset in shifts:
        offset_x, offset_y = offset
        mask_shifted = mask.copy()
        
        #ensures the shift is within height and width of image
        mask_shifted = mask_shifted[
            max(0, offset_y): min(h, h + offset_y),
            max(0, offset_x): min(w, w + offset_x)
        ]
        #needed padding to restore original image height and width
        padding = [
            (max(0, -offset_y), max(0, offset_y)),
            (max(0, -offset_x), max(0, offset_x))
        ]
        mask_shifted = np.pad(mask_shifted, padding)
        mask = np.clip(mask_shifted + mask, 0, 1) # ensures combined mask values remain within [0,1]
    print('Done expanding text mask')
    return mask


def get_text_mask(image, coordinates_to_mask):   
    text_mask = np.zeros_like(image[:, :, 0])
    for coordinates in coordinates_to_mask:
        xmin, ymin, xmax, ymax = coordinates
        bbox = image[ymin : ymax, xmin : xmax, :]
        white_text = (bbox > 250).all(axis=-1)
        text_mask[ymin : ymax, xmin : xmax] = white_text
    
    expanded_mask = get_mask_expanded(text_mask, 5)
    image[expanded_mask == 1] = 0
    expanded_mask *= 255
    print('Done text mask')
    return expanded_mask


def get_image_inpainted(image, image_mask):
    # Perform image inpainting to remove text from the original image
    image_inpainted = cv2.inpaint(
        image, image_mask, inpaintRadius=7, flags=cv2.INPAINT_NS
    )

    return image_inpainted

def clean_image(image_path, cleaned_dir, ocr_dir):
    im = cv2.imread(image_path)
    image_name = os.path.basename(image_path)
    
    os.makedirs(ocr_dir, exist_ok=True)
    base_name = os.path.splitext(image_name)[0]
    image_ocr_path = os.path.join(ocr_dir, f"{base_name}.json")
    coordinates = get_meme_text(im, image_ocr_path)
    im_mask = get_text_mask(image=im, coordinates_to_mask=coordinates)

    # (DEBUG) Read/write mask file to check 
    # cv2.imwrite(f"mask_{image_name}", im_mask)
    # im_mask = cv2.imread(f"mask_{image_name}", cv2.IMREAD_GRAYSCALE)
    os.makedirs(cleaned_dir, exist_ok=True)

    # Perform image inpainting
    im_inpainted = get_image_inpainted(image=im, image_mask=im_mask)

    cv2.imwrite(f"{cleaned_dir}/{image_name}", im_inpainted)

    print('Done inpainting', image_path)

################### ENHANCE IMAGE ##########################

def enhance_image(image_path, enhanced_dir, color=1., contrast=1., sharpness=1., brightness=1.):
    """
    Slightly enhance the image by adjusting its contrast, sharpness, and brightness.
    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path to save the enhanced image.
        color/contrast/sharpness/brightness (float): Factor by which to enhance the respective. Default to unchanged (1.0).
            - color: 0.0 gives black and white, >1.0 gives higher color saturation
            - contrast: 0.0 gives solid gray, >1.0 gives higher contrast
            - brightness: 0.0 gives black, >1.0 gives higher brightness
            - sharpness: 0.0 gives blurred, >1.0 gives more sharpened image
    """
    image = Image.open(image_path)
    
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(color)
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(sharpness)
    
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)
    
    os.makedirs(enhanced_dir, exist_ok=True)
    image.save(f"{enhanced_dir}/{os.path.basename(image_path)}")

################### TEXT TRANSLATION ##########################

def get_correct_ocr(image_ocr_path):
    """
    OCR text may not be good, and common autocorrect libraries can not address this well.
    Ideally use LLM, but for now let's use dictionary.
    """
    with open(image_ocr_path, 'r') as f:
        ocr_data = json.load(f)
    texts = ocr_data.keys()
    ocr_data_corrected = {}
    for t in texts:
        corrected_t = t
        for wrong, correct in correction_dict.items():
            corrected_t = corrected_t.replace(wrong, correct)
        ocr_data_corrected[corrected_t] = ocr_data[t]
    return ocr_data_corrected

def split_text_into_lines(text, coordinates):
    words = text.split()
    num_lines = len(coordinates)
    avg_words_per_line = len(words) // num_lines
    
    widths = [xmax - xmin for xmin, ymin, xmax, ymax in coordinates]
    total_width = sum(widths)
    
    lines = [''] * num_lines
    word_idx = 0
    
    for i in range(num_lines):
        while word_idx < len(words):
            line_width_ratio = widths[i] / total_width
            num_words = round(line_width_ratio * len(words))
            if word_idx + num_words > len(words) or i == num_lines - 1:
                lines[i] = ' '.join(words[word_idx:])
                break
            lines[i] = ' '.join(words[word_idx:word_idx + num_words])
            word_idx += num_words
            if len(lines[i]) > 0:
                break
            
    return lines


def get_translated_ocr(image_ocr_path):
    """
    Usually text in memes are split into multiple lines, better concatenate to full sentence for correct translation.
    If one word a line, might be addressing a person rather than a sentence. 
    Parameters:
        image_ocr_path (str): Path to the ocr text and coordinates json. Dictionary with text as key and coordinates as values.
        correction_dict (dict): Keys are wrong word/syllable, values are corresponding correct one.
    Returns:
        a dictionary of translated ocr with corresponding coordinates
    """   
    ocr_data = get_correct_ocr(image_ocr_path)
    translated_ocr_data = {}
    concatenated_text = ""
    spread_coordinates = []  # of concatenated_text
    
    for t in ocr_data.keys():
        if len(t.split()) == 1:
            translated_text = translator.translate(t, dest='vi').text
            translated_ocr_data[translated_text] = ocr_data[t]
        elif len(t.split()) > 1:
            concatenated_text += " " + t
            spread_coordinates.append(ocr_data[t])
    
    if concatenated_text:
        concatenated_text = concatenated_text.strip()
        translated_text = translator.translate(concatenated_text, dest='vi').text
        lines = split_text_into_lines(translated_text, spread_coordinates)
        
        for line, coord in zip(lines, spread_coordinates):
            translated_ocr_data[line] = coord
    
    return translated_ocr_data

      
def translate_image(image_path, image_ocr_path, translated_dir):
    os.makedirs(translated_dir, exist_ok=True)
    image = Image.open(image_path)
    ocr_data = get_translated_ocr(image_ocr_path)
    draw = ImageDraw.Draw(image)
    font = font_manager.FontProperties(family='sans-serif', weight='bold')
    file = font_manager.findfont(font)
    # draw.font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)
    
    for text, coords in ocr_data.items(): 
        xmin, ymin, xmax, ymax = coords
        draw.font = ImageFont.truetype(file,35)
        draw.text((xmin, ymin), text, fill="white")
    image.save(f"{translated_dir}/{os.path.basename(image_path)}")
    
if __name__ == "__main__":
    image_type = ('.png','.jpg','.jpeg')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['clean','enhance','translate'])
    
    parser.add_argument('--img_dir', type=str, help='Relative path to raw image folder',required=False, default='img/')
    parser.add_argument('--cleaned_dir', type=str, help='Relative path to directory text-removed images',required=False, default='img_cleaned/')
    parser.add_argument('--enhanced_dir', type=str, help='Relative path to enhanced image folder',required=False, default='img_enhanced/')
    parser.add_argument('--ocr_dir', type=str, help='Relative path to folder storing ocr text and coordinates',required=False, default='ocr/')
    parser.add_argument('--translated_dir', type=str, help='Relative path to translated image folder',required=False, default='img_translated/')

 
    args = parser.parse_args()
    
    img_dir = os.path.abspath(args.img_dir)
    cleaned_dir = os.path.abspath(args.cleaned_dir)
    enhanced_dir = os.path.abspath(args.enhanced_dir)
    ocr_dir = os.path.abspath(args.ocr_dir)
    translated_dir = os.path.abspath(args.translated_dir)
    # parse arguments    
    if args.mode=='clean':
        reader = easyocr.Reader(['en'])    
        print("Cleaning images at:", img_dir)
        for filename in os.listdir(img_dir):
            if filename.lower().endswith(image_type):
                image_path = os.path.join(img_dir, filename)
                clean_image(image_path, cleaned_dir, ocr_dir)
        print("Cleaned and save images to:", cleaned_dir)
        
    elif args.mode=='enhance':
        for filename in os.listdir(cleaned_dir):
            if filename.lower().endswith(image_type):
                image_path = os.path.join(cleaned_dir, filename)
                enhance_image(image_path, enhanced_dir, color=1.15, contrast=1.15, sharpness=1.15, brightness=1.15)
        print('Enhanced and saved enhanced images to:', enhanced_dir)
    
    elif args.mode=='translate':
        correction_dict = {"[":'I', 'WASA': 'WAS A'}
        translator = Translator()
        for filename in os.listdir(cleaned_dir):
            if filename.lower().endswith(image_type):
                image_path = os.path.join(cleaned_dir, filename)
                base_name = os.path.splitext(filename)[0]
                image_ocr_path = os.path.join(ocr_dir, f"{base_name}.json")
                translate_image(image_path, image_ocr_path, translated_dir)
        print('Translated and saved images to:', translated_dir)
        