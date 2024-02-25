import os

import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm

from config import lung_input_size, model_path, lung_model_name, UPLOAD_FOLDER, voxel_max, voxel_min
from model import build_unet
from preprocess_data import intensity_normalize
from utils import load_weights, bbox_location, split_mask
import tensorflow as tf
import nibabel as nib


def lung_segment(image, file_name):
    img_shape = image.shape
    model = build_unet(lung_input_size, chanel=32, stage=5)
    # model.summary()
    load_weights(model, model_path, lung_model_name)
    lung_mask = lung_predict(model, img_shape, file_name)
    lung_mask = possprocess(image, lung_mask)

    mean = np.mean(image[(lung_mask > 0).astype(int) > 0])
    std = np.std(image[(lung_mask > 0).astype(int) > 0])

    max_value = max(np.percentile(image[(lung_mask > 0).astype(int) > 0], 95), voxel_max)
    min_value = max(np.percentile(image[(lung_mask > 0).astype(int) > 0], 5), voxel_min)

    norm_image = intensity_normalize(image, max_value, min_value, mean, std)

    return norm_image, lung_mask


def predict(image, model):
    img = tf.image.resize(image, (224, 224))
    shape = image.shape
    # img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    mask = model.predict(img[tf.newaxis, ...], verbose=0)
    # mask = mask.astype(float)
    mask = np.where(mask > 0.5, 1, 0).astype(int)

    mask = tf.image.resize(mask, (shape[0], shape[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.squeeze(mask, -1)

    return mask[0]


def lung_predict(model, img_shape, file_name):
    mask = np.zeros(img_shape)
    index = 0
    print("Lung segment")
    for i in tqdm(range(len(os.listdir(os.path.join(UPLOAD_FOLDER, "image_elevation"))))):
        # print(os.path.join(UPLOAD_FOLDER, f"image_elevation/{file}"))

        img = cv2.imread(os.path.join(UPLOAD_FOLDER, f"image_elevation/{file_name}_{i}.png"))
        # cv2.imshow("im", img)
        # cv2.waitKey(0)
        img_mask = predict(img, model)
        mask[:, :, index] = img_mask
        index += 1
    return mask


def post_process(mask):
    # Remove noise
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
