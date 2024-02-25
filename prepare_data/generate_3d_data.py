import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from numpy import float16
from scipy.ndimage import label

from config import train_dir, train_lung_mask_dir, voxel_max, median_spacing, voxel_min, images_3d_path, mask_3d_path
from preprocess_data import preprocess
from utils import bbox_location, bbox_2D, square_image_and_mask, square_3D_image_and_mask
from visualize import multi_slice_viewer


def generate_label(mask):
    shape = mask.shape
    new_mask = np.zeros(shape, dtype=float16)
    s = np.ones((3, 3, 3))
    labels, num_components = label(mask, structure=s)

    for i in range(1, num_components + 1):
        label_mask = (labels == i).astype(np.uint8)
        area = np.sum(label_mask)

        if area > 100:
            value = 1 - np.sqrt(area / np.sum(np.ones_like(mask)))
            new_mask += label_mask * value

    return new_mask


def save_data(images, masks, ggo_mask, file_name, min_value):
    # print(file_name)
    if np.sum(ggo_mask) > 100:
        bbox = bbox_location(masks)

        # print(bbox)
        _bbox = bbox_location(ggo_mask)

        width = bbox[3] - bbox[2]
        x_center = int((bbox[3] + bbox[2]) / 2)
        padding = min(int(width * 0.2), int(32))

        if x_center - padding - 128 < 0:
            x_center = padding + 128

        if x_center + padding + 128 > 512:
            x_center = 512 - padding - 128

        # global count
        for i in range(_bbox[4], _bbox[5] + 1, 12):

            start = i if i + 12 < bbox[5] + 1 else bbox[5] - 11
            finish = i + 12 if i + 12 < bbox[5] + 1 else bbox[5] + 1

            if np.sum(ggo_mask[:, :, start:finish]) > 20:

                for j in range(3):
                    if width > 256:
                        start_col = x_center - 128 + padding * (j - 1)
                        end_col = start_col + 256

                        if np.sum(ggo_mask[bbox[0]:bbox[1], start_col:end_col, start:finish]) > 20:
                            sq_img, sq_mask = square_3D_image_and_mask(
                                images[bbox[0]:bbox[1], start_col:end_col, start:finish],
                                ggo_mask[bbox[0]:bbox[1], start_col:end_col, start:finish],
                                min_value)

                            if np.sum(sq_mask) < 100:
                                continue

                            sq_mask = generate_label(sq_mask)
                            if np.any(sq_mask):
                                np.save(os.path.join(images_3d_path, f"{file_name}_{i}_{j}.npy"), sq_img)
                                np.save(os.path.join(mask_3d_path, f"{file_name}_{i}_{j}.npy"), sq_mask)
                    else:
                        if j != 1:

                            start_col = x_center - 128 + padding * (j - 1)
                            end_col = start_col + 256

                            if np.sum(ggo_mask[bbox[0]:bbox[1], start_col:end_col, start:finish]) > 20:
                                sq_img, sq_mask = square_3D_image_and_mask(
                                    images[bbox[0]:bbox[1], start_col:end_col, start:finish],
                                    ggo_mask[bbox[0]:bbox[1], start_col:end_col, start:finish],
                                    min_value)

                                sq_mask = generate_label(sq_mask)

                                if np.any(sq_mask):
                                    np.save(os.path.join(images_3d_path, f"{file_name}_{i}_{j}.npy"), sq_img)
                                    np.save(os.path.join(mask_3d_path, f"{file_name}_{i}_{j}.npy"), sq_mask)


for file in os.listdir(train_lung_mask_dir):
    file_name = file.split(".")[0]
    print(file_name)
    image_load = nib.load(os.path.join(train_dir, f"{file_name}_ct.nii.gz"))
    image = image_load.get_fdata()
    spacing = image_load.header.get_zooms()[:3]
    ggo_mask = nib.load(os.path.join(train_dir, f"{file_name}_seg.nii.gz")).get_fdata()

    lung_mask = nib.load(os.path.join(train_lung_mask_dir, file)).get_fdata()

    lung_mask = np.round(lung_mask)

    mean = np.mean(image[((lung_mask > 0).astype(int) - ggo_mask) > 0])
    std = np.std(image[((lung_mask > 0).astype(int) - ggo_mask) > 0])

    max_value = max(np.percentile(image[((lung_mask > 0).astype(int) - ggo_mask) > 0], 95), voxel_max)
    min_value = max(np.percentile(image[((lung_mask > 0).astype(int) - ggo_mask) > 0], 5), voxel_min)

    new_images, new_ggo_masks, new_lung_mask = preprocess(image, max_value, min_value, mean, std, ggo_mask, lung_mask)

    upper_mask = (new_lung_mask == 10).astype(int)
    lower_mask = (new_lung_mask == 20).astype(int)

    min_value = np.amin(new_images[np.nonzero(new_lung_mask)])
    upper_ggo_mask = np.logical_and(upper_mask, new_ggo_masks)
    lower_ggo_mask = np.logical_and(lower_mask, new_ggo_masks)

    # multi_slice_viewer(image, upper_ggo_mask)
    # plt.show()

    save_data(new_images, upper_mask, upper_ggo_mask, f"{file_name}_{i}_up", min_value)
    save_data(new_images, lower_mask, lower_ggo_mask, f"{file_name}_{i}_low", min_value)
