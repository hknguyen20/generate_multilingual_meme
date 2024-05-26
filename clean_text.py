import cv2
import numpy as np
import easyocr
from PIL import Image

reader = easyocr.Reader(['en'])

def get_meme_text(image):
    coordinates = []
    full = reader.readtext(image)
    text = reader.readtext(image, detail = 0, paragraph=True)
    print(full)
    for coord, _,_ in full:
        xmin = min([p[0] for p in coord])
        xmax = max([p[0] for p in coord])
        ymin = min([p[1] for p in coord])
        ymax = max([p[1] for p in coord])
        coordinates.append((xmin, ymin, xmax, ymax))
    print('done ocr')
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
    
    expanded_mask = expand_text_mask(text_mask, 4)
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

if __name__ == "__main__":   
    # image_path = '/home/ubuntu/shared/ospc/img/0002.png'
    image_path = '/home/ubuntu/20nguyen.hk/test_imgs/english.png'
    im = cv2.imread(image_path)
    cv2.imwrite("original_image.png", im)
    text, coordinates = get_meme_text(image=im)
    print(text)
    im_mask = get_text_mask(image=im, coordinates_to_mask=coordinates)

    # (OPTIONAL) If necessary to read/write to /tmp file system for reading later
    # Write to /tmp folder
    cv2.imwrite("temp_image_mask.png", im_mask)

    # (OPTIONAL) Read from /tmp folder
    im_mask = cv2.imread("temp_image_mask.png", cv2.IMREAD_GRAYSCALE)

    # Perform image inpainting
    im_inpainted = get_image_inpainted(image=im, image_mask=im_mask)

    cv2.imwrite("temp_image_inpainted.png", im_inpainted)

    print('done inpainting')
 