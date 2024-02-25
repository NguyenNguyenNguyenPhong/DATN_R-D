import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
from scipy.ndimage import label, binary_closing
from tqdm import tqdm

import utils, visualize
from config import val_dir, lggo_pred_dir, sggo_pred_dir, val_lung_pred_dir, merge_dir


def post_process_2d(image, mask, lung_mask, lggo_mask):
    intersect_mask = np.logical_and(mask, lggo_mask)

    s = np.ones((3, 3, 3))
    slabels, snum_components = label(mask, structure=s)

    selem = np.ones((3, 3, 3), dtype=np.bool)
    closed_image = binary_closing(lggo_mask, structure=selem)

    llabels, lnum_components = label(closed_image, structure=s)

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

check = False
for file in os.listdir(val_lung_pred_dir):
    file_name = file.split(".")[0]
    print(file_name)

    # if file_name == "radiopaedia_10_85902_3":
    #     check = True
    #
    # if not check:
    #     continue

    image = nib.load(os.path.join(val_dir, file))
    sggo_mask = nib.load(os.path.join(sggo_pred_dir, file)).get_fdata()
    lggo_mask = nib.load(os.path.join(lggo_pred_dir, file)).get_fdata()
    lung_mask = nib.load(os.path.join(val_lung_pred_dir, file)).get_fdata()

    result = post_process_2d(image.get_fdata(), sggo_mask, (lung_mask > 0).astype(int), lggo_mask)

    # visualize.multi_slice_viewer(result, image.get_fdata())
    #
    # plt.show()

    seg = nib.Nifti1Image(result, image.affine, image.header)

    nib.save(seg, os.path.join(merge_dir, file))
