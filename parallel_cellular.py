# ------------------------------------------------------------
# Parallel Cellular Algorithm (Matrix Iteration Output Only)

# Includes USN: 1BM23CS239
# ------------------------------------------------------------

import numpy as np
import cv2
from google.colab import files

USN = "1BM23CS239"
print("Parallel Cellular Automata – Matrix Iteration Output")
print("USN:", USN)
print("------------------------------------------------------\n")

# ------------------------------------------------------------
# Upload image
# ------------------------------------------------------------
uploaded = files.upload()
img_path = list(uploaded.keys())[0]

# Load grayscale image
img = cv2.imread(img_path, 0)
img = cv2.resize(img, (100, 100))   # matrix easier to view

# ------------------------------------------------------------
# Define PCA Update Rule
# ------------------------------------------------------------
def pca_update(image):
    padded = np.pad(image, 1, mode='edge')
    new_img = np.zeros_like(image)

    for i in range(1, padded.shape[0]-1):
        for j in range(1, padded.shape[1]-1):

            center = padded[i, j]
            neighbors = [
                padded[i-1,j-1], padded[i-1,j], padded[i-1,j+1],
                padded[i,j-1],                 padded[i,j+1],
                padded[i+1,j-1], padded[i+1,j], padded[i+1,j+1]
            ]

            diff = np.mean([abs(center - n) for n in neighbors])

            # PCA rule: edge enhancement or smoothing
            if diff > 20:
                new_img[i-1,j-1] = min(center + diff, 255)
            else:
                new_img[i-1,j-1] = max(center - 5, 0)

    return new_img

# ------------------------------------------------------------
# Perform 6 iterations and print matrices
# ------------------------------------------------------------
iterations = 6
matrix = img.copy()

for t in range(iterations):
    print(f"\nIteration {t+1} Matrix:")
    print(matrix)
    matrix = pca_update(matrix)

print("\nProcessing Completed Successfully for USN:", USN)
output:
Parallel Cellular Automata – Matrix Iteration Output
USN: 1BM23CS239
------------------------------------------------------

WhatsApp Image 2025-11-18 at 11.54.08 PM (2).jpeg
WhatsApp Image 2025-11-18 at 11.54.08 PM (2).jpeg(image/jpeg) - 166565 bytes, last modified: 18/11/2025 - 100% done
Saving WhatsApp Image 2025-11-18 at 11.54.08 PM (2).jpeg to WhatsApp Image 2025-11-18 at 11.54.08 PM (2).jpeg

Iteration 1 Matrix:
[[194 196 197 ...  73  79  76]
 [199 201 148 ... 144 143 138]
 [195 201 192 ... 141 125 107]
 ...
 [164 156 186 ... 180 182 182]
 [161 155 185 ... 182 183 182]
 [157 151 182 ... 182 184 183]]
/tmp/ipython-input-238255951.py:43: RuntimeWarning: overflow encountered in scalar subtract
  diff = np.mean([abs(center - n) for n in neighbors])

Iteration 2 Matrix:
[[255 255 255 ... 205 153 211]
 [255 196 255 ... 205 207 201]
 [255 196 255 ... 255 255 255]
 ...
 [159 255 181 ... 255 215 215]
 [227 255 233 ... 246 215 255]
 [221 255 227 ... 255 179 247]]

Iteration 3 Matrix:
[[250 250 250 ... 255 255 206]
 [250 255 255 ... 255 255 255]
 [250 255 255 ... 250 255 255]
 ...
 [255 255 255 ... 255 255 255]
 [255 255 255 ... 255 255 255]
 [255 255 255 ... 255 255 255]]

Iteration 4 Matrix:
[[255 255 255 ... 250 250 255]
 [255 250 250 ... 250 250 250]
 [255 250 250 ... 255 250 250]
 ...
 [250 250 250 ... 250 250 250]
 [250 250 250 ... 250 250 250]
 [250 250 250 ... 250 250 250]]

Iteration 5 Matrix:
[[250 250 250 ... 245 255 250]
 [250 255 255 ... 255 255 255]
 [250 255 255 ... 250 255 255]
 ...
 [245 255 255 ... 255 245 245]
 [245 245 255 ... 255 245 245]
 [245 245 255 ... 255 245 245]]

Iteration 6 Matrix:
[[255 255 255 ... 255 250 255]
 [255 250 250 ... 250 250 250]
 [255 250 250 ... 255 250 250]
 ...
 [255 250 250 ... 250 255 240]
 [255 255 250 ... 250 255 240]
 [240 255 250 ... 250 255 240]]

Processing Completed Successfully for USN: 1BM23CS239
