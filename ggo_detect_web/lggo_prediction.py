import os

import nibabel
import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter
from tqdm import tqdm

from config import model_path, sggo_best_fold, sggo_model_name, lggo_input_size, lggo_best_fold, lggo_model_name, \
    layer_3d_number
from model import LGGO_Segment
from utils import bbox_location, erode_mask, load_weights, gaussian_weighting
import tensorflow as tf

models = []


def load_model(name, fold=None):
    if fold is not None:
        segment_model_name = f"{name}_fold_{fold}"
        lggo = LGGO_Segment(lggo_input_size, channel=32, stage=5)
        model = lggo.unet
        model = load_weights(model, model_path, segment_model_name)
        models.append(model)

    else:
        for i in range(5):
            segment_model_name = f"{name}_fold_{i}"
            lggo = LGGO_Segment(lggo_input_size, channel=32, stage=5)
            model = lggo.unet
            model = load_weights(model, model_path, segment_model_name)
            if model != 0:
                models.append(model)


def predict(model, image, len_model, voxel_ratio):
    im_shape = image.shape
    image = tf.image.resize(image, (256, 256))
    input_image = tf.expand_dims(image, axis=-1)

    mask = model.predict(input_image[tf.newaxis, ...], verbose=0)

    mask = tf.squeeze(mask, -1)

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

    squared_img = np.ones((size, size, layer_3d_number)) * min_val

    if image.shape[0] > image.shape[1]:
        squared_img[:, dif: dif + image.shape[1], :] = image

    else:
        squared_img[dif: dif + image.shape[0], :, :] = image

    squared_masks = []
    square_gaussians = []

    for model in models:
        squared_mask, square_gaussian = predict(model, squared_img, len(models), voxel_ratio)

        squared_masks.append(squared_mask)
        square_gaussians.append(square_gaussian)

    org_masks = []
    org_gaussian = []

    for squared_mask, square_gaussian in zip(squared_masks, square_gaussians):
        if image.shape[0] > image.shape[1]:
            org_masks.append(squared_mask[:, dif: dif + image.shape[1], :])
            org_gaussian.append(square_gaussian[:, dif: dif + image.shape[1], :])
        else:
            org_masks.append(squared_mask[dif: dif + image.shape[0], :, :])
            org_gaussian.append(square_gaussian[dif: dif + image.shape[0], :, :])

    return org_masks, org_gaussian


def split_and_predict(image, lung_mask, models, min_val, voxel_ratio):
    bbox = bbox_location(lung_mask)
    masks = []
    gaussians = []
    for i in range(len(models)):
        masks.append(np.zeros(image.shape))
        gaussians.append(np.ones(image.shape) * 1e-6)
    width = bbox[3] - bbox[2]
    finish = 0

    x_center = int((bbox[3] + bbox[2]) / 2)
    padding = min(int(width * 0.2), int(image.shape[0] / 4))

    if x_center - padding - 128 < 0:
        x_center = padding + 128

    if x_center + padding + 128 > image.shape[0]:
        x_center = image.shape[0] - padding - 128

    for i in tqdm(range(bbox[4], bbox[5] + 1, int(layer_3d_number / 2))):
        # print(i)
        if width > 256:
            for j in range(3):
                start = i if i + layer_3d_number < bbox[5] + 1 else bbox[5] - layer_3d_number + 1
                finish = i + layer_3d_number if i + layer_3d_number < bbox[5] + 1 else bbox[5] + 1

                start_col = x_center - 128 + padding * (j - 1)
                end_col = start_col + 256

                img = image[bbox[0]:bbox[1], start_col:end_col, start:finish]

                pred_masks, gaussian_values = square_3d_image(img, min_val, models, voxel_ratio)
                for k in range(len(masks)):
                    # print(pred_masks[k].shape, gaussian_values[k].shape)
                    masks[k][bbox[0]:bbox[1], start_col:end_col, start:finish] += pred_masks[k] * gaussian_values[k]
                    gaussians[k][bbox[0]:bbox[1], start_col:end_col, start:finish] += gaussian_values[k]

        else:
            start = i if i + layer_3d_number < bbox[5] + 1 else bbox[5] - layer_3d_number + 1
            finish = i + layer_3d_number if i + layer_3d_number < bbox[5] + 1 else bbox[5] + 1

            start_col = x_center - 128
            end_col = start_col + 256

            img = image[bbox[0]:bbox[1], start_col:end_col, start:finish]

            pred_masks, gaussian_values = square_3d_image(img, min_val, models, voxel_ratio)
            for k in range(len(masks)):
                # print(pred_masks[k].shape, gaussian_values[k].shape)
                masks[k][bbox[0]:bbox[1], start_col:end_col, start:finish] += pred_masks[k] * gaussian_values[k]
                gaussians[k][bbox[0]:bbox[1], start_col:end_col, start:finish] += gaussian_values[k]

                # print(gaussian_values)

    ensemble_mask = np.zeros(image.shape)

    for i in range(len(masks)):
        masks[i][bbox[0]:bbox[1], x_center - 128 - padding:x_center + 128 + padding, bbox[4]:finish] /= gaussians[i][
                                                                                                            bbox[0]:
                                                                                                            bbox[1],
                                                                                                            x_center - 128 - padding:x_center + 128 + padding,
                                                                                                            bbox[4]:finish]
        ensemble_mask += (masks[i] > 0.5).astype(int)

    return (ensemble_mask >= ((len(models) - 1) / 2)).astype(int)


def lggo_segment(image, lung_mask, voxel_ratio, best_model=False):
    name = lggo_model_name

    load_model(name, None if not best_model else lggo_best_fold)

    lung_mask = np.round(lung_mask)

    upper_mask = (lung_mask == 10).astype(int)
    upper_mask = erode_mask(upper_mask)
    lower_mask = (lung_mask == 20).astype(int)
    lower_mask = erode_mask(lower_mask)

    min_value = np.amin(image[np.nonzero(lung_mask)])

    print("Lggo upper")
    upper_lggo_mask = split_and_predict(image, upper_mask, models, min_value, voxel_ratio)
    print("Lggo lower")
    lower_lggo_mask = split_and_predict(image, lower_mask, models, min_value, voxel_ratio)

    mask = np.logical_or(upper_lggo_mask, lower_lggo_mask)

    return mask


if __name__ == "__main__":
    image = nibabel.load("static/upload/volume-covid19-A-0026_ct.nii.gz").get_fdata()

    lung_mask = nibabel.load(
        "D:\\Github\\data\\COVID-19-20\\COVID-19-20_v2\\new_lung_test_mask\\volume-covid19-A-0026.nii.gz").get_fdata()

    lggo_segment(image, lung_mask, 6, True)
