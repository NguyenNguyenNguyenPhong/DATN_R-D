import os
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import dice_coefficient, hausdorff_distance_3d, surface_dice_3d, true_negative_rate, true_positive_rate, \
    lggo_post_process
from visualize import multi_slice_viewer

pred_mask_dir = "DBL_CBL"
groud_truth_mask_dir = "test_mask"

if __name__ == "__main__":
    sum_dsc = 0
    sum_hausdorff = 0
    sum_NSD = 0
    sum_sensitivity = 0
    sum_specificity = 0

    for file in tqdm(os.listdir(pred_mask_dir)):
        test_mask = nib.load(os.path.join(pred_mask_dir, file)).get_fdata()

        gtr_mask = nib.load(os.path.join(groud_truth_mask_dir, file)).get_fdata()

        voxel_spacing = nib.load(os.path.join(pred_mask_dir, file)).header.get_zooms()

        test_mask = lggo_post_process(test_mask, voxel_spacing[-1]/voxel_spacing[0])

        sum_dsc += dice_coefficient(gtr_mask, (test_mask > 0).astype(int))
        sum_hausdorff += hausdorff_distance_3d(gtr_mask, (test_mask > 0).astype(int), voxel_spacing, 95)
        sum_NSD += surface_dice_3d(gtr_mask, test_mask)
        sum_sensitivity += true_positive_rate(gtr_mask, (test_mask > 0).astype(int))
        sum_specificity += true_negative_rate(gtr_mask, (test_mask > 0).astype(int))

    print("DSC:", sum_dsc/20)
    print("Hausdorff:", sum_hausdorff/20)
    print("NSD:", sum_NSD/20)
    print("sensitivity:", sum_sensitivity/20)
    print("specificity:", sum_specificity/20)
