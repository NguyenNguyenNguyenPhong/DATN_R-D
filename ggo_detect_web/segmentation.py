import os

import cv2
import numpy as np
from scipy.ndimage import label, binary_closing
from tqdm import tqdm

import utils
from lggo_prediction import lggo_segment
from lung_prediction import lung_segment
from sggo_prediction import sggo_segment
from utils import resize_3d_image_cubic, norm_image, recreate_directory
import nibabel as nib


def merge_ggo(image, mask, lung_mask, lggo_mask):
    intersect_mask = np.logical_and(mask, lggo_mask)

    s = np.ones((3, 3, 3))
    slabels, snum_components = label(mask, structure=s)

    selem = np.ones((3, 3, 3), dtype=np.bool)
    # closed_image = binary_closing(lggo_mask, structure=selem)

    llabels, lnum_components = label(lggo_mask, structure=s)

    hard_ggo = np.zeros(mask.shape)

    mean_value = np.mean(image[(lung_mask - np.logical_or(mask, lggo_mask)) > 0])

    std = np.std(image[(lung_mask - np.logical_or(mask, lggo_mask)) > 0])

    for i in tqdm(range(1, snum_components + 1)):
        _label = (slabels == i).astype(int)
        _bbox = utils.bbox_location(_label)

        if np.sum(np.logical_and(_label[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]],
                                 intersect_mask[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]])) > 0:
            hard_ggo[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]] = np.logical_or(
                _label[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]],
                hard_ggo[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]])

        elif np.sum(np.logical_and(_label[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]],
                                   lung_mask[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]])) > 20:
            if np.mean(image[np.nonzero(_label)]) > mean_value - 2 * std:
                hard_ggo[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]] = np.logical_or(
                    hard_ggo[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]],
                    _label[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]])

    for i in tqdm(range(1, lnum_components + 1)):
        _label = (llabels == i).astype(int)

        _bbox = utils.bbox_location(_label)

        if np.sum(np.logical_and(_label[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]],
                                 intersect_mask[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]])) > 0:
            hard_ggo[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]] = np.logical_or(
                _label[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]],
                hard_ggo[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]])

        elif np.sum(np.logical_and(_label[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]],
                                   lung_mask[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]])) > 20:
            if np.mean(image[np.nonzero(_label)]) > mean_value - 2 * std:
                hard_ggo[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]] = np.logical_or(
                    hard_ggo[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]],
                    _label[_bbox[0]: _bbox[1], _bbox[2]:_bbox[3], _bbox[4]:_bbox[5]])

    return hard_ggo


def draw_contour(image, mask, file_name, folder):
    recreate_directory(folder)
    for layer in range(image.shape[2]):
        # Extract the current layer from the image and mask
        image_layer = image[:, :, layer]
        img = norm_image(image_layer)
        mask_layer = mask[:, :, layer]

        # Find contours in the mask for the current layer
        contours, _ = cv2.findContours(mask_layer.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the image for the current layer
        image_with_contours = img.copy()
        cv2.drawContours(image_with_contours, contours, -1, (255, 0, 0),
                         1)  # You can change the color and thickness here

        # Save the image with contours to a file
        layer_file_name = os.path.join(folder, f"{file_name}_{layer}.png")
        cv2.imwrite(layer_file_name, image_with_contours)


def do_segmentation(image, file_name, voxel_ratio, ensemble=False):
    norm_img, lung_mask = lung_segment(image, file_name)

    sggo_mask = sggo_segment(norm_img, lung_mask, not ensemble)
    lggo_mask = lggo_segment(norm_img, lung_mask, voxel_ratio, not ensemble)

    mask = merge_ggo(image, sggo_mask, lung_mask, lggo_mask)

    resize_img = resize_3d_image_cubic(image)
    resize_mask = resize_3d_image_cubic(mask)

    trans_img = np.transpose(resize_img, (2, 0, 1))
    trans_mask = np.transpose(resize_mask, (2, 0, 1))

    trans_img = np.flip(trans_img, 0)
    trans_mask = np.flip(trans_mask, 0)

    draw_contour(image, mask, file_name, "static/lung_segment/image_elevation")
    draw_contour(trans_img, trans_mask, file_name, "static/lung_segment/image_side")

    return file_name, resize_img.shape[-1], trans_img.shape[-1]


if __name__ == "__main__":
    image = nib.load("static/upload/volume-covid19-A-0187_ct.nii.gz").get_fdata()
    do_segmentation(image, "volume-covid19-A-0187_ct", 1)
