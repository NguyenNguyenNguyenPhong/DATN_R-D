import os

import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import label, generate_binary_structure

from utils.visualize import multi_slice_viewer

lung_mask_dir = "new_lung_test_mask"
lung_process_mask_dir = "D:/Github/data/COVID-19-20/COVID-19-20_v2/add_lung_test_mask"


def smooth_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for i in range(mask.shape[-1]):
        img = mask[:, :, i]
        if np.sum(img) > 10:
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            mask[:, :, i] = img

    return mask


def post_process(mask):
    labels, num_labels = ndimage.label(mask)

    # Calculate the size (volume) of each connected component
    sizes = ndimage.sum(mask, labels, range(1, num_labels + 1))

    # Find the indices of the 2 largest connected components
    largest_components = np.argsort(sizes)[::-1][:2]

    # Create a bounding box that contains the 2 largest components
    components_mask = np.zeros_like(mask)
    for label in largest_components:
        components_mask[labels == (label + 1)] = 1
        box_indices = np.where(components_mask)
        x_min, y_min, z_min = np.min(box_indices, axis=1)
        x_max, y_max, z_max = np.max(box_indices, axis=1)
        if x_min < 150 / 512 * mask.shape[0] and x_max > 350 / 512 * mask.shape[0]:
            break

    return smooth_mask(components_mask)


def smooth_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for i in range(mask.shape[-1]):
        img = mask[:, :, i]
        if np.sum(img) > 10:
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            mask[:, :, i] = img

    return mask


def possprocess(image, lung_mask):
    # Keep largest component 3D
    mask = post_process(lung_mask)

    bbox = bbox_location(mask)

    new_lung_mask = np.zeros(mask.shape)

    for i in range(bbox[4], bbox[5] + 1):
        # print(i)
        new_lung_mask[:, :, i] = split_mask(image[:, :, i], mask[:, :, i], bbox)

    return new_lung_mask


if __name__ == '__main__':
    check = True
    for file in os.listdir(lung_mask_dir):
        print(file)
        # if file == "volume-covid19-A-0073.nii.gz":
        #     check = True

        if check:
            mask_nib = nib.load(os.path.join(lung_mask_dir, file))

            mask = mask_nib.get_fdata()

            processed_mask = possprocess(mask)

            # multi_slice_viewer(mask, processed_mask)
            #
            # plt.show()

            # print(processed_mask.shape)

            seg = nib.Nifti1Image(processed_mask, mask_nib.affine, mask_nib.header)

            nib.save(seg, os.path.join(lung_process_mask_dir, file))

        # break_flag
