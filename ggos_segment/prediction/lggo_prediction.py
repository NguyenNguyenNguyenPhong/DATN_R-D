import os

import tensorflow as tf
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
from numpy import uint8, float16
from tqdm import tqdm

from arch.lggo_segment import LGGO_Segment
from config import input_3d_shape, voxel_max, voxel_min, num_layer
from train.call_back import load_weights
from utils.preprocess_data import preprocess, return_original, intensity_normalize
from utils.utils import bbox_location, compute_gaussian, gaussian_weighting
from utils.visualize import multi_slice_viewer

segment_model_folder = "D:\\Github\\ggos_segmentation\\train\\weight\\ggo"

models = []


def load_model(name, fold=None):
    if fold is not None:
        segment_model_name = f"{name}_fold_{fold}"
        lggo = LGGO_Segment(input_3d_shape, channel=32, stage=5)
        model = lggo.unet
        load_weights(model, segment_model_folder, segment_model_name)
        models.append(model)

    else:
        for i in range(5):
            segment_model_name = f"{name}_fold_{i}"
            print(segment_model_name)
            lggo = LGGO_Segment(input_3d_shape, channel=32, stage=5)
            model = lggo.unet
            model = load_weights(model, segment_model_folder, segment_model_name)
            if model != 0:
                models.append(model)


def predict(model, image, voxel_ratio, len_model):
    im_shape = image.shape
    image = tf.image.resize(image, (256, 256))
    input_image = tf.expand_dims(image, axis=-1)

    mask = model.predict(input_image[tf.newaxis, ...], verbose=0)

    mask = tf.cast(tf.squeeze(mask, -1), tf.float16)

    mask = tf.image.resize(mask, (im_shape[0], im_shape[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if len_model == 1:
        # print("Gaussian weighting")
        weight = gaussian_weighting(mask[0], sigma=1. / 8, threshold=0.4, voxel_ratio=voxel_ratio)
    else:
        weight = np.ones(im_shape)

    return mask[0], weight


def square_3d_image(image, min_val, models, voxel_ratio):
    size = np.amax(image.shape)
    dif = int((np.amax(image.shape[0:2]) - np.amin(image.shape[0:2])) / 2)

    # print(dif)
    squared_img = np.ones((size, size, num_layer)) * min_val

    if image.shape[0] > image.shape[1]:
        squared_img[:, dif: dif + image.shape[1], :] = image

    else:
        squared_img[dif: dif + image.shape[0], :, :] = image

    squared_masks = []
    square_gaussians = []

    for model in models:
        squared_mask, square_gaussian = predict(model, squared_img, voxel_ratio, len(models))

        squared_masks.append(squared_mask)
        square_gaussians.append(square_gaussian)

    org_masks = []
    org_gaussian = []

    for squared_mask, square_gaussian in zip(squared_masks, square_gaussians):
        if image.shape[0] > image.shape[1]:
            org_masks.append(squared_mask[:, dif: dif + image.shape[1], :])
            org_gaussian.append(square_gaussian[:, dif: dif + image.shape[1], :])
            # multi_slice_viewer(image, squared_mask[:, dif: dif + image.shape[1], :])
        else:
            org_masks.append(squared_mask[dif: dif + image.shape[0], :, :])
            # print(np.unique(squared_mask[dif: dif + image.shape[0], :, :]))
            org_gaussian.append(square_gaussian[dif: dif + image.shape[0], :, :])
            # multi_slice_viewer(image, squared_mask[dif: dif + image.shape[0], :, :])

        # plt.show()

    return org_masks, org_gaussian


def split_and_predict(image, lung_mask, models, min_val, voxel_ratio):
    bbox = bbox_location(lung_mask)
    masks = []
    gaussians = []
    for i in range(len(models)):
        masks.append(np.zeros(image.shape, dtype=float16))
        gaussians.append(np.zeros(image.shape, dtype=float16))
    width = bbox[3] - bbox[2]
    finish = 0

    x_center = int((bbox[3] + bbox[2]) / 2)
    padding = min(int(width * 0.2), int(image.shape[0] / 4))

    if x_center - padding - 128 < 0:
        x_center = padding + 128

    if x_center + padding + 128 > image.shape[0]:
        x_center = image.shape[0] - padding - 128

    for i in tqdm(range(bbox[4], bbox[5] + 1, int(num_layer/2))):
        for j in range(3):
            start = i if i + num_layer < bbox[5] + 1 else bbox[5] - num_layer + 1
            finish = i + num_layer if i + num_layer < bbox[5] + 1 else bbox[5] + 1

            start_col = x_center - 128 + padding * (j - 1)
            end_col = start_col + 256

            img = image[bbox[0]:bbox[1], start_col:end_col, start:finish]

            pred_masks, gaussian_values = square_3d_image(img, min_val, models, voxel_ratio)
            for k in range(len(masks)):
                # print(pred_masks[k].shape, gaussian_values[k].shape)
                masks[k][bbox[0]:bbox[1], start_col:end_col, start:finish] += pred_masks[k] * gaussian_values[k]
                gaussians[k][bbox[0]:bbox[1], start_col:end_col, start:finish] += gaussian_values[k]

                # print(gaussian_values)

    ensemble_mask = np.zeros(image.shape, dtype=uint8)

    for i in range(len(masks)):
        masks[i][bbox[0]:bbox[1], x_center - 128 - padding:x_center + 128 + padding, bbox[4]:finish] /= gaussians[i][
                                                                                                        bbox[0]:bbox[1],
                                                                                                        x_center - 128 - padding:x_center + 128 + padding,
                                                                                                        bbox[4]:finish]
        ensemble_mask += (masks[i] > 0.5).astype(uint8)

    return (ensemble_mask > ((len(models) - 1) / 2)).astype(uint8)


lung_mask_dir = "D:\\Github\\data\\COVID-19-20\\COVID-19-20_v2\\new_lung_test_mask"
val_dir = "D:\\Github\\data\\COVID-19-20\\COVID-19-20_v2\\Validation"
save_dir = "D:\\Github\\data\\COVID-19-20\\COVID-19-20_v2\\3D_result\\test_DFL_ensemble"

if __name__ == "__main__":

    check = False

    model_name = "LGGO_segment_32_5_IN_ATT_DFL"
    load_model(model_name, 4)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for file in os.listdir(lung_mask_dir):
        file_name = file.split(".")[0]
        print(file_name)

        if file_name == "volume-covid19-A-0026":
            check = True

        if not check:
            continue

        org_image = nib.load(os.path.join(val_dir, f"{file_name}_ct.nii.gz")).get_fdata()
        lung_mask_load = nib.load(os.path.join(lung_mask_dir, file))

        voxel_spacing = lung_mask_load.header.get_zooms()
        voxel_ratio = voxel_spacing[-1] / voxel_spacing[0]

        lung_mask = lung_mask_load.get_fdata()

        mean = np.mean(org_image[lung_mask > 0])
        std = np.std(org_image[lung_mask > 0])
        max_value = max(np.percentile(org_image[lung_mask > 0], 95), voxel_max)
        min_value = max(np.percentile(org_image[lung_mask > 0], 5), voxel_min)

        image = intensity_normalize(org_image, max_value, min_value, mean, std)

        lung_mask = np.round(lung_mask)

        upper_mask = (lung_mask == 10).astype(uint8)
        lower_mask = (lung_mask == 20).astype(uint8)

        min_value = np.amin(image[np.nonzero(lung_mask_dir)])

        # mask = np.zeros(lung_mask.shape)

        print("Lggo upper")
        upper_lggo_mask = split_and_predict(image, upper_mask, models, min_value, voxel_ratio)
        print("Lggo lower")
        lower_lggo_mask = split_and_predict(image, lower_mask, models, min_value, voxel_ratio)

        mask = np.logical_or(upper_lggo_mask, lower_lggo_mask).astype(uint8)

        multi_slice_viewer(image, np.logical_or(upper_lggo_mask, lower_lggo_mask).astype(int))
        plt.show()

        # seg = nib.Nifti1Image(mask, lung_mask_load.affine, lung_mask_load.header)
        # #
        # nib.save(seg, os.path.join(save_dir, file))
