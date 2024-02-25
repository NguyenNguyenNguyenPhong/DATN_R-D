import random

import cv2
import numpy as np
import os

bimap_dir = 'annotations/bitmaps'

file_names = []
for file in os.listdir(bimap_dir):
    file_names.append(file.split(".")[0])
    mask = cv2.imread(os.path.join(bimap_dir, file), 0)
    bimap = np.where(mask > 0, 1, 0)
    cv2.imwrite(os.path.join(bimap_dir, file), bimap)

random.shuffle(file_names)

# Calculate the split index based on a 7:3 ratio
split_idx = int(0.7 * len(file_names))

# Split the file names into train/val and test sets
trainval_files = file_names[:split_idx]
test_files = file_names[split_idx:]

# Write the file names to the trainval.txt file, one per line
with open('annotations/trainval.txt', 'w') as f:
    for file_name in trainval_files:
        f.write(file_name + '\n')

# Write the file names to the test.txt file, one per line
with open('annotations/test.txt', 'w') as f:
    for file_name in test_files:
        f.write(file_name + '\n')