import os

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from utils import find_and_fill_contours, interpolate_img, fill_convex, bbox_2D, bbox_location, split_mask, filter_mask
from visualize import multi_slice_viewer

if __name__ == "__main__":
    lung_mask_dir = "add_lung_test_mask"
    train_dir = "test"
    new_lung_mask_dir = "new_add_lung_mask"

    check = True

    for file in os.listdir(lung_mask_dir):
        file_name = file.split(".")[0]
        print(file_name)
        # if file_name == "radiopaedia_27_86410_0":
        #     check = True

        if check:
            img = nib.load(os.path.join(train_dir, file))
            lung_mask = nib.load(os.path.join(lung_mask_dir, file))
            # ggo_mask = nib.load(os.path.join(train_dir, f"{file_name}_seg.nii.gz")).get_fdata()
            img_data = img.get_fdata()
            lung_mask_data = lung_mask.get_fdata()

            lung_mask_data = filter_mask(lung_mask_data)

            # multi_slice_viewer(img_data, lung_mask_data)
            # plt.show()

            bbox = bbox_location(lung_mask_data)

            new_lung_mask = np.zeros(lung_mask_data.shape)

            for i in range(bbox[4], bbox[5] + 1):
                # print(i)
                new_lung_mask[:, :, i] = split_mask(img_data[:, :, i], (lung_mask_data[:, :, i] > 0).astype(int), bbox)

            # multi_slice_viewer(img_data, new_lung_mask)
            # #
            # plt.show()

            seg = nib.Nifti1Image(new_lung_mask, lung_mask.affine, lung_mask.header)

            nib.save(seg, os.path.join(new_lung_mask_dir, file))
