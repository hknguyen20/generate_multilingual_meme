from PIL import Image, ImageEnhance
import os
import argparse

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

    
if __name__ == "__main__":
    image_type = ('.png','.jpg','.jpeg')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cleaned_dir', type=str, help='Relative path to clean images folder',required=False, default='img_cleaned/')
    parser.add_argument('--enhanced_dir', type=str, help='Relative path to raw image folder',required=True)
    
    args = parser.parse_args()
    cleaned_dir = os.path.abspath(args.cleaned_dir)
    enhanced_dir = os.path.abspath(args.enhanced_dir)
    # parse arguments    
    for filename in os.listdir(cleaned_dir):
        if filename.lower().endswith(image_type):
            image_path = os.path.join(cleaned_dir, filename)
            enhance_image(image_path, enhanced_dir)
    print('Enhanced and saved enhanced images to:', enhanced_dir)
