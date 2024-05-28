import cv2
import numpy as np
import easyocr
import os
import argparse

reader = easyocr.Reader(['en'])

def get_meme_text(image):
    coordinates = []
    full = reader.readtext(image)
    text = reader.readtext(image, detail = 0, paragraph=True)
    for coord, _,_ in full:
        xmin = round(min([p[0] for p in coord])) #round to integer for slicing later
        xmax = round(max([p[0] for p in coord]))
        ymin = round(min([p[1] for p in coord]))
        ymax = round(max([p[1] for p in coord]))
        coordinates.append((xmin, ymin, xmax, ymax))
    print('Done ocr')
    print(coordinates)
    print(text)
    return text, coordinates

def expand_text_mask(mask, shift_amount):
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
    
    expanded_mask = expand_text_mask(text_mask, 3)
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

def process_image(image_path, cleaned_dir):
    im = cv2.imread(image_path)
    image_name = os.path.basename(image_path)
    text, coordinates = get_meme_text(image=im)
    im_mask = get_text_mask(image=im, coordinates_to_mask=coordinates)

    # (DEBUG) Read/write mask file to check 
    # cv2.imwrite(f"mask_{image_name}", im_mask)
    # im_mask = cv2.imread(f"mask_{image_name}", cv2.IMREAD_GRAYSCALE)
    os.makedirs(cleaned_dir, exist_ok=True)

    # Perform image inpainting
    im_inpainted = get_image_inpainted(image=im, image_mask=im_mask)

    cv2.imwrite(f"{cleaned_dir}/{image_name}", im_inpainted)

    print('Done inpainting', image_path)

if __name__ == "__main__":
    image_type = ('.png','.jpg','.jpeg')
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, help='Relative path to raw image folder',required=True)
    parser.add_argument('--cleaned_dir', type=str, help='Relative path to directory storing text-removed images',required=False, default='img_cleaned/')
    
    args = parser.parse_args()
    img_dir = os.path.abspath(args.img_dir)
    cleaned_dir = os.path.abspath(args.cleaned_dir)
    # parse arguments    
    print("Cleaning images at:", img_dir)
    for filename in os.listdir(img_dir):
        if filename.lower().endswith(image_type):
            image_path = os.path.join(img_dir, filename)
            process_image(image_path, cleaned_dir)
    print("Cleaned and save images to:", cleaned_dir)

        