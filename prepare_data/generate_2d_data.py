import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from numpy import float16
from scipy.ndimage import label
from config import train_dir, voxel_max, voxel_min, images_2d_path, mask_2d_path
from preprocess_data import intensity_normalize
from utils import bbox_location, bbox_2D, square_image_and_mask, erode_mask
from visualize import multi_slice_viewer

sggo_path = "dataset/labelsTr_3"
lggo_path = "dataset/labelsTr_2"
lung_mask_dir = "new_lung_train_mask"


def generate_label(mask):
    shape = mask.shape
    new_mask = np.zeros(shape, dtype=float16)
    labels, num_components = label(mask)

    for i in range(1, num_components + 1):
        label_mask = (labels == i).astype(np.uint8)
        area = np.sum(label_mask)
        if area > 20:
            value = 1 - np.sqrt(area / np.sum(np.ones_like(mask)))

            new_mask += label_mask * value

    return new_mask


def save_data(images, masks, ggo_mask, file_name, min_value):
    bbox = bbox_location(masks)

    for i in range(bbox[4], bbox[5] + 1):
        start_col = bbox[2]
        end_col = bbox[3]

        if np.sum(ggo_mask[bbox[0]:bbox[1], start_col:end_col, i]) > 20:
            _bbox = bbox_2D(masks[:, :, i])
            sq_img, sq_mask = square_image_and_mask(images[bbox[0]:bbox[1], start_col:end_col, i],
                                                    ggo_mask[bbox[0]:bbox[1], start_col:end_col, i], min_value)

            sq_mask = generate_label(sq_mask)

            np.save(os.path.join(images_2d_path, f"{file_name}_{i}.npy"), sq_img)
            np.save(os.path.join(mask_2d_path, f"{file_name}_{i}.npy"), sq_mask)


for file in os.listdir(lung_mask_dir):
    file_name = file.split(".")[0]
    print(file_name)
    image = nib.load(os.path.join(train_dir, f"{file_name}_ct.nii.gz")).get_fdata()
    ggo_mask = nib.load(os.path.join(train_dir, f"{file_name}_seg.nii.gz")).get_fdata()
    lung_mask = nib.load(os.path.join(lung_mask_dir, file)).get_fdata()

    mean = np.mean(image[((lung_mask > 0).astype(int) - ggo_mask) > 0])
    std = np.std(image[((lung_mask > 0).astype(int) - ggo_mask) > 0])

    max_value = max(np.percentile(image[((lung_mask > 0).astype(int) - ggo_mask) > 0], 95), voxel_max)
    min_value = max(np.percentile(image[((lung_mask > 0).astype(int) - ggo_mask) > 0], 5), voxel_min)

    image, ggo_mask = intensity_normalize(image, max_value, min_value, mean, std, ggo_mask)

    lung_mask = np.round(lung_mask)

    upper_mask = (lung_mask == 10).astype(int)
    upper_mask = erode_mask(upper_mask)
    lower_mask = (lung_mask == 20).astype(int)
    lower_mask = erode_mask(lower_mask)

    # multi_slice_viewer(image, upper_mask)
    # plt.show()

    min_value = np.amin(image[np.nonzero(lung_mask_dir)])
    upper_seg_image = np.multiply(image, upper_mask) + np.logical_not(upper_mask) * min_value
    upper_ggo_mask = np.logical_and(upper_mask, ggo_mask)
    lower_seg_image = np.multiply(image, lower_mask) + np.logical_not(lower_mask) * min_value
    lower_ggo_mask = np.logical_and(lower_mask, ggo_mask)
    #
    # bbox = bbox_location(upper_seg_image)

    save_data(image, upper_mask, upper_ggo_mask, f"{file_name}_up", min_value)
    save_data(image, lower_mask, lower_ggo_mask, f"{file_name}_low", min_value)

    # break
