# Multilingual Memes Generator

---

## Introduction
The spread of harmful memes is a growing and pressing issue on online platforms. Competitions such as the [Online Safety Prize Challenge](https://ospc.aisingapore.org/) and the [Hateful Memes Challenge](https://ai.meta.com/blog/hateful-memes-challenge-and-data-set/) highlight the significance of this field. Building robust detection models is crucial for effectively mitigating hate speech and promoting online safety. While there are some English-based meme datasets, there is a lack of datasets in low-resource settings, such as Vietnam. Translating English datasets to Vietnamese can be one approach to addressing this gap.

## Contribution
- **Image Cleaning Method**: A simple, lightweight, yet effective method for cleaning images, useful for applications such as object detection, feature extraction, captioning, etc.
- **Multilingual Meme Generation**: A method for generating multilingual memes from existing English-based datasets. Although currently limited to Vietnamese, this method can be easily adapted to all other languages supported by Google Translate.

## Technical Overview
This project is lightweight and simple to understand and use:

1. **Remove Text from Image**
   - Detect OCR text using `easyOCR`.
   - Identify and mask white text.
   - Inpaint the masked areas using `CV2`.

2. **Translate Image**
   - Translate detected OCR text using Google Translate (`googletrans`).
   - Write the translated text back onto the image at the correct coordinates.

## How to Install
#### Clone the repo
```bash
git clone https://github.com/hknguyen20/generate_multilingual_meme.git
cd generate_multilingual_meme/
```
#### Set up virtual environment and install dependencies
**Important:** Ensure you have Python3.9
```bash
bash setup.sh
```

## Usage
**Important:** Ensure you clean the text from the images before translating.

### Clean Text
To clean the text from the images, use:
```bash
python main.py --mode=clean
```
You can view the cleaned images in the `img_cleaned` folder.

### Enhance Images (Optional)
If desired, you can enhance the images to make them more vivid:
```bash
python main.py --mode=enhance
```
You can view the enhanced images in the `img_enhanced` folder.

### Translate Text
Finally, to translate the text in the images, use:
```bash
python main.py --mode=translate
```
If you want to translate text on enhanced images, use:
```bash
python main.py --mode=translate --cleaned_dir=img_enhanced/
```
You can view the translated memes in the `img_translated` folder.


## Limitations
- **Capitalized Meme Text Only**: `easyOCR` detects capitalized words well but struggles with non-capitalized words. Consequently, translations may be poor for non-capitalized text.
- **White Meme Text Without Thick Borders**: The logic for detecting text pixels assumes the text is white. It will not work well for memes with text in other colors or those with thick borders and shadows.
