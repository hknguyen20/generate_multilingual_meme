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
## Usage section
## Limitations
