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

def apply_shifts(mask, shift_amount):
    """
    Apply shifts in various directions to the mask to cover slight misalignments.
    """
    #horizontal, vertical, diagonal shifts
    shifts = [(offset_x, offset_y) for offset_x in (0, shift_amount, -shift_amount) 
                   for offset_y in (0, shift_amount, -shift_amount)]

    final_mask = np.zeros_like(mask)
    for offset_x, offset_y in shifts:
        shifted_mask = np.roll(mask, shift=(offset_x, offset_y), axis=(0, 1))
        final_mask = np.clip(final_mask + shifted_mask, 0, 1)
    return final_mask


def get_text_mask(image, coordinates_to_mask):
    # Create a mask image with image_size
    text_mask = np.zeros_like(image[:, :, 0])
    for coordinates in coordinates_to_mask:
        xmin, ymin, xmax, ymax = coordinates
        bbox = image[ymin : ymax, xmin : xmax, :]
        white_text = (bbox > 250).all(axis=-1)
        text_mask[ymin : ymax, xmin : xmax] = white_text
    
    shifted_mask = apply_shifts(text_mask, 5)
    image[shifted_mask == 1] = 0
    shifted_mask *= 255
    print('done text mask')
    return shifted_mask


def get_image_inpainted(image, image_mask):
    # Perform image inpainting to remove text from the original image
    image_inpainted = cv2.inpaint(
        image, image_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA
    )

    return image_inpainted

if __name__ == "__main__":   
    # image_path = '/home/ubuntu/shared/ospc/img/0002.png'
    image_path = '/home/ubuntu/20nguyen.hk/ospc/img/english.png'
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
 