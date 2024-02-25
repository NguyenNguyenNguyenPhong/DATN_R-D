import os

import numpy as np
from tqdm import tqdm

from config import model_path, sggo_input_size, sggo_best_fold, sggo_model_name
from model import SGGO_Segment
from utils import bbox_location, erode_mask, load_weights
import tensorflow as tf

models = []


def load_model(name, fold=None):
    if fold is not None:
        segment_model_name = f"{name}_{fold}_Att"
        ggo = SGGO_Segment(sggo_input_size, channel=32, stage=5)
        model = ggo.unet
        load_weights(model, model_path, segment_model_name)
        models.append(model)

    else:
        for i in range(5):
            segment_model_name = f"{name}_{i}_Att"
            ggo = SGGO_Segment(sggo_input_size, channel=32, stage=5)
            model = ggo.unet
            load_weights(model, model_path, segment_model_name)
            models.append(model)


def predict(models, image):
    shape = image.shape
    input_image = tf.expand_dims(image, axis=-1)
    input_image = tf.image.resize(input_image, (256, 256))

    mask = np.zeros(input_image.shape)
    mask = mask[tf.newaxis, ...]
    for model in models:
        _mask = model.predict(input_image[tf.newaxis, ...], verbose=0)
        mask += (_mask > 0.5).astype(int)

    mask = (mask > 2).astype(int) if len(models) > 1 else mask
    mask = tf.image.resize(mask, shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    mask = tf.squeeze(mask, -1)

    return mask[0]


def square_2d_image(image, min_val, models):
    size = np.amax(image.shape)
    # print(image.shape)
    dif = int((np.amax(image.shape[0:2]) - np.amin(image.shape[0:2])) / 2)
    # print(dif)
    squared_img = np.ones((size, size)) * min_val

    if image.shape[0] > image.shape[1]:
        squared_img[:, dif: dif + image.shape[1]] = image

    else:
        squared_img[dif: dif + image.shape[0], :] = image

    norm_square_img = np.ones((256, 256)) * min_val
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

    for i in tqdm(range(bbox[4], bbox[5] + 1)):
        start_col = bbox[2]
        end_col = bbox[3]

        # Use variables in indexing
        img = image[bbox[0]:bbox[1], start_col:end_col, i]
        pred_mask = square_2d_image(img, min_val, models)

        mask[bbox[0]:bbox[1], start_col:end_col, i] = pred_mask

    return mask


def sggo_segment(image, lung_mask, best_model=False):
    name = sggo_model_name
    load_model(name, None if not best_model else sggo_best_fold)

    lung_mask = np.round(lung_mask)

    upper_mask = (lung_mask == 10).astype(int)
    upper_mask = erode_mask(upper_mask)
    lower_mask = (lung_mask == 20).astype(int)
    lower_mask = erode_mask(lower_mask)

    min_value = np.amin(image[np.nonzero(lung_mask)])

    print("Sggo upper")
    upper_lggo_mask = split_and_predict(image, upper_mask, models, min_value)
    print("Sggo lower")
    lower_lggo_mask = split_and_predict(image, lower_mask, models, min_value)

    mask = np.logical_or(upper_lggo_mask, lower_lggo_mask)

    return mask
