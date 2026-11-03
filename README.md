# Melanoma Busters

OpenCV skin-lesion analysis using the dermatology **ABCD rule**: Asymmetry, Border, Color and Diameter. It comes with a command-line tool and a live web interface where you can upload an image, use your webcam or pick a sample.

> ⚠️ **Educational / research tool only. Not a medical diagnosis.** See a dermatologist for any skin concern.

## Quick start

Requires Python 3 with OpenCV and NumPy:

```bash
pip install opencv-python numpy

# Web interface -> http://localhost:8000
python3 server.py

# Command line
python3 melanoma.py samples/malignant_ISIC_0000022.jpg
python3 melanoma.py image.jpg --px-per-mm 10 --no-show   # mm diameter, save results only
```

## Pipeline

1. Resize the image and convert it to **grayscale**
2. Remove hair (black-hat morphology, then inpainting)
3. Blur to reduce noise, then boost contrast with CLAHE
4. Detect the dermoscope's black lens frame and ignore it
5. Otsu threshold and morphological cleanup to find the lesion; keep the most central blob
6. Check that the region is a plausible mole; otherwise report **"Mole not detected"**
7. Extract the **contour**, convex hull and fitted ellipse
8. Compute the ABCD features:
   - **A**: asymmetry, measured by mirroring the lesion across its two principal axes
   - **B**: border irregularity (compactness and solidity)
   - **C**: number of reference dermoscopic colors present
   - **D**: maximum diameter (in mm if a scale is given)
9. Combine them into the Stolz TDS score `1.3A + 0.1B + 0.5C + 0.5D`: below 4.75 is benign-looking, above 5.45 is highly suspicious

## Limitations

- Rule-based thresholds, tested on only 16 images (about 75% agreement with the labels)
- Grayscale segmentation struggles with pink, non-pigmented lesions
- Not validated for clinical use

## Sample data

The images in `samples/` come from the [ISIC Archive](https://www.isic-archive.com) and are licensed CC-0. Their labels (biopsy-confirmed) are in `samples/labels.csv`.
