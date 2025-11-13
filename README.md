# DevDigitizer Project (Sanskrit OCR)

## Updates - Sanskrit OCR Laboratory Project
**Under:** Prof. Navneet Goyal  
**By:** Naman Goenka  
**Date:** December 2020

### Key Improvements

**1. Segmentation Improvements**
- Fixed segmentation off-by-one errors in letter/word coordinate extraction which was resulting in the last line being ignored.
- Segmentation robust to page layout.

**2. Lexical Correction (Levenshtein Distance)**
- Post-OCR correction using dynamic programming-based edit distance
- Dictionary: 9,311 Sanskrit words (ITRANS format) open sourced
- Configurable threshold (default: max distance = 2)
- Integrated into text conversion pipeline

**3. Image Preprocessing for Degraded Manuscripts**
- Data augmentation: noise, blur, skew, perspective transforms
- Enhancement: denoising, sharpening, adaptive thresholding
- Lightweight super-resolution: bicubic upscaling with unsharp masking

**4. Dataset Augmentation Using Bhagavad Gita**
- Automated pipeline: ITRANS extraction → Devanagari rendering → letter segmentation
- Source: sanskritdocuments.org (701 verses, 10 processed for demonstration)
- Generated 447 labeled letter images across 102 original classes
- Module: `gita_dataset_augmentation/`

**5. Evaluation Framework**
- WER (Word Error Rate) calculation with/without lexical correction
- Per-image and aggregate metrics
- Script: `evaluate_wer_on_images.py`

#### Course System Issues: 
- Added automatic cleanup of temporary directories (tmp_words, tmp_letters)
- Added gitignore and requirements file for more reproducable setup

#### Files Added
- `utils/lexicalcorrector.py` - Levenshtein distance correction
- `utils/image_preprocessor.py` - Image enhancement utilities
- `evaluate_wer_on_images.py` - WER evaluation script
- `gita_dataset_augmentation/` - Complete dataset generation pipeline
- `requirements.txt` - All dependencies

---

## About

The DevDigitizer project aims to build a state of the art Optical Character Recognition Software for Sanskrit/ Samskritam (Devanagari Script). The project is commited to developing novel document analysis, computer vision, deep learning and search algorithms through persistent research, inorder to build a robust and highly accurate Sanskrit OCR system.  

**The**  **Vision**  of the DevDigitizer project is to facilitate digitization and preservation of ancient indian texts on Science, Math, Literature, Poetry etc... written in Sanskrit (Devanagari Script). Digitization of ancient Manuscripts will increase the ease of access to these documents for further research and study.

---

# Installation

The Software is currently being refactored for public use and will be made availble for use very soon.

# Python Dependencies

1. Numpy
2. Tensorflow
3. Keras
4. OpenCV
5. Flask 

# Sanskrit Letter Dataset 
The dataset used for this work is available in the following github repo :
https://github.com/avadesh02/Sanskrit-letter-dataset/blob/master/README.md

# Citing DevDigitizer

If you want to cite **DevDigitizer** in your papers, please use the following bibtex line:

<cite> Avadesh, Meduri, and Navneet Goyal. "Optical Character Recognition for Sanskrit Using Convolution Neural Networks." In 2018 13th IAPR International Workshop on Document Analysis Systems (DAS), pp. 447-452. IEEE, 2018. </cite> 
