import os

import tensorflow as tf
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from scipy.ndimage import label

from arch.ggo_model import build_unet
from config import voxel_max, voxel_min
from utils.preprocess_data import intensity_normalize
from utils.utils import bbox_location, erode_mask, compute_gaussian
from utils.visualize import multi_slice_viewer

segment_model_folder = "D:\\Github\\ggos_segmentation\\train\\weight\\ggo"

input_shape = (256, 256, 1)

models = []


def load_model(name, fold=None):
    if fold is not None:
        print("Load")
        print(fold)
        segment_model_name = f"{name}_fold_{fold}"
        print(segment_model_name)
        model = build_unet(input_shape, 32, 6)

        model.load_weights(os.path.join(segment_model_folder, segment_model_name))
        models.append(model)

    else:
        for i in range(5):
            segment_model_name = f"{name}_fold_{i}"
            print(segment_model_name)
            model = build_unet(input_shape, 32, 6)
            model.load_weights(os.path.join(segment_model_folder, segment_model_name))
            models.append(model)


def predict(models, image):
    shape = image.shape
    input_image = tf.expand_dims(image, axis=-1)
    input_image = tf.image.resize(input_image, (256, 256))

    mask = np.zeros(input_image.shape)
    mask = mask[tf.newaxis, ...]
    for model in models:
        _mask = model.predict(input_image[tf.newaxis, ...])
        mask += (_mask > 0.5).astype(int)

    # mask = (mask > 0.5).astype(int)
    mask = tf.image.resize(mask, shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    mask = tf.squeeze(mask, -1)

    return mask[0]


def square_2d_image(image, min_val, models):
    size = np.amax(image.shape)
    # print(image.shape)
    dif = int((np.amax(image.shape[0:2]) - np.amin(image.shape[0:2])) / 2)
    # print(dif)
    squared_img = np.ones((size, size)) * (min_val - 1)

    if image.shape[0] > image.shape[1]:
        squared_img[:, dif: dif + image.shape[1]] = image

    else:
        squared_img[dif: dif + image.shape[0], :] = image

    norm_square_img = np.ones((256, 256)) * (min_val - 1)
    extend = int((256 - size) / 2)

    if size < 256:
        norm_square_img[extend: extend + size, extend: extend + size] = squared_img
        norm_square_mask = predict(models, norm_square_img)
        squared_mask = norm_square_mask[extend: extend + size, extend: extend + size]

    else:
        squared_mask = predict(models, squared_img)

    if image.shape[0] > image.shape[1]:
        return squared_mask[:, dif: dif + image.shape[1]]

    else:
        return squared_mask[dif: dif + image.shape[0], :]


def split_and_predict(image, lung_mask, models, min_val):
    mask = np.zeros(image.shape)
    bbox = bbox_location(lung_mask)

    height = bbox[1] - bbox[0]

    width = bbox[3] - bbox[2]

    for i in range(bbox[4], bbox[5] + 1):
        start_col = bbox[2]
        end_col = bbox[3]

        # Use variables in indexing
        img = image[bbox[0]:bbox[1], start_col:end_col, i]
        pred_mask = square_2d_image(img, min_val, models)

        mask[bbox[0]:bbox[1], start_col:end_col, i] = pred_mask

    return mask


lung_mask_dir = "D:\\Github\\data\\COVID-19-20\\COVID-19-20_v2\\new_add_lung_mask"
val_dir = "D:\\Github\\data\\COVID-19-20\\COVID-19-20_v2\\test"
save_dir = "D:\\Github\\data\\COVID-19-20\\COVID-19-20_v2\\2D_result\\test_CBL_fold_1"
image_dir = "D:\\Github\\data\\COVID-19-20\\COVID-19-20_v2\\detection_data\\images"

if __name__ == "__main__":
    name = "SGGO_segment_32_5_CBL"
    load_model(name, fold=1)
    # check = False
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for file in os.listdir(lung_mask_dir):
        file_name = file.split(".")[0]

        print(file_name)
        image = nib.load(os.path.join(val_dir, f"{file_name}.nii.gz")).get_fdata()
        lung_mask_load = nib.load(os.path.join(lung_mask_dir, file))

        lung_mask = lung_mask_load.get_fdata()

        mean = np.mean(image[lung_mask > 0])
        std = np.std(image[lung_mask > 0])
        max_value = max(np.percentile(image[lung_mask > 0], 95), voxel_max)
        min_value = max(np.percentile(image[lung_mask > 0], 5), voxel_min)

        image = intensity_normalize(image, max_value, min_value, mean, std)

        lung_mask = np.round(lung_mask)

        upper_mask = (lung_mask == 10).astype(int)
        upper_mask = erode_mask(upper_mask)
        lower_mask = (lung_mask == 20).astype(int)
        lower_mask = erode_mask(lower_mask)

        min_value = np.amax(image[np.nonzero(lung_mask_dir)])

        voxel_spacing = lung_mask_load.header.get_zooms()

        upper_lggo_mask = split_and_predict(image, upper_mask, models, min_value)
        lower_lggo_mask = split_and_predict(image, lower_mask, models, min_value)

        mask = np.logical_or(upper_lggo_mask, lower_lggo_mask)

        print(np.sum(mask))

        seg = nib.Nifti1Image(mask, lung_mask_load.affine, lung_mask_load.header)

        nib.save(seg, os.path.join(save_dir, file))

        # multi_slice_viewer(image, mask)
        # plt.show()
