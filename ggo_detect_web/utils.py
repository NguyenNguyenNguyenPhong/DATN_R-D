import os
import shutil

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import int16, uint8, float16
from scipy.ndimage import label, gaussian_filter, zoom
from typing import Union, Tuple, List

from skimage.measure import regionprops


def filter_mask(mask):
    # label the connected components in the mask
    new_mask = np.zeros(mask.shape)
    labeled_mask, num_cc = label(mask)

    for i in range(1, num_cc + 1):
        label_mask = (labeled_mask == i).astype(int)
        if np.sum(label_mask) < 20:
            continue

        new_mask += label_mask

    return new_mask


def bbox_location(mask):
    non_zero = np.where(mask)
    x_min, y_min, z_min = np.min(non_zero, axis=1)
    x_max, y_max, z_max = np.max(non_zero, axis=1)

    return x_min, x_max, y_min, y_max, z_min, z_max


def find_and_fill_contours(image):
    # Find contours in the binary image
    plt.imsave("tmp.png", image, cmap="gray")
    img = cv2.imread("tmp.png", 0)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image for filling contours
    filled_image = np.zeros(image.shape)

    # Fill contours on the image
    cv2.drawContours(filled_image, contours, -1, 1, thickness=cv2.FILLED)

    return filled_image


def fill_convex(img):
    plt.imsave("tmp.png", img, cmap="gray")
    img = cv2.imread("tmp.png", 0)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw the filled convex hull
    mask = np.zeros_like(img)

    # Iterate over each contour and find the convex hull
    for contour in contours:
        hull = cv2.convexHull(contour)
        cv2.fillConvexPoly(mask, hull, 1)

    return mask


def resize_mask(image, shape):
    img = np.zeros(shape)
    im_shape = image.shape
    dif_x = int(np.abs(im_shape[1] - shape[1]) / 2)
    dif_y = int(np.abs(im_shape[0] - shape[0]) / 2)
    if im_shape[0] < shape[0] or im_shape[1] < shape[1]:
        # print("!!!")
        img[dif_y:im_shape[0] + dif_y, dif_x:im_shape[1] + dif_x] = image

    else:
        img = image[dif_y:shape[0] + dif_y, dif_x:shape[1] + dif_x]

    return img


def interpolate_img(img, scale):
    plt.imsave("tmp.png", img, cmap="gray")
    img = cv2.imread("tmp.png", 0)
    resized_image = cv2.resize(img, None, fx=scale, fy=scale)
    # plt.imshow(resized_image)
    # plt.show()

    return resize_mask(resized_image, img.shape)


def smooth_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for i in range(mask.shape[-1]):
        img = mask[:, :, i]
        plt.imsave("tmp.png", img, cmap="gray")
        img = cv2.imread("tmp.png", 0)
        if np.sum(img) > 10:
            # img = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            mask[:, :, i] = img

    return (mask > 0).astype(int)


def erode_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for i in range(mask.shape[-1]):
        img = mask[:, :, i]
        plt.imsave("tmp.png", img, cmap="gray")
        img = cv2.imread("tmp.png", 0)
        if np.sum(img) > 10:
            # img = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.erode(img, kernel)
            mask[:, :, i] = img

    return (mask > 0).astype(int)


def bbox_2D(mask):
    image1_non_zero = np.nonzero(mask)
    x_min = np.amin(image1_non_zero[1])
    x_max = np.amax(image1_non_zero[1])

    y_min = np.amin(image1_non_zero[0])
    y_max = np.amax(image1_non_zero[0])

    return x_min, x_max, y_min, y_max, (x_max - x_min) * (y_max - y_min)


def square_image_and_mask(img, mask, min_value):
    size = np.amax(img.shape)
    dif = int((np.amax(img.shape) - np.amin(img.shape)) / 2)
    squared_img = np.ones((size, size)) * (min_value - 1)
    squared_mask = np.zeros((size, size))

    if img.shape[0] > img.shape[1]:
        squared_img[:, dif: dif + img.shape[1]] = img
        squared_mask[:, dif: dif + img.shape[1]] = mask

    else:
        squared_img[dif: dif + img.shape[0], :] = img
        squared_mask[dif: dif + img.shape[0], :] = mask

    norm_square_img = np.ones((256, 256)) * (min_value - 1)
    norm_square_mask = np.zeros((256, 256))
    extend = int((256 - size) / 2)

    if size < 256:
        norm_square_img[extend: extend + size, extend: extend + size] = squared_img
        norm_square_mask[extend: extend + size, extend: extend + size] = squared_mask

        return norm_square_img, norm_square_mask

    return squared_img, squared_mask


def square_3D_image_and_mask(img, mask, min_value):
    size = np.amax(img.shape[:2])
    dif = int((np.amax(img.shape[:2]) - np.amin(img.shape[:2])) / 2)
    squared_img = np.ones((size, size, 12), dtype=float16) * (min_value)
    squared_mask = np.zeros((size, size, 12))

    if img.shape[0] > img.shape[1]:
        squared_img[:, dif: dif + img.shape[1], :] = img
        squared_mask[:, dif: dif + img.shape[1], :] = mask

    else:
        squared_img[dif: dif + img.shape[0], :, :] = img
        squared_mask[dif: dif + img.shape[0], :, :] = mask

    norm_square_img = np.ones((256, 256, 12), dtype=float16) * (min_value)
    norm_square_mask = np.zeros((256, 256, 12))
    extend = int((256 - size) / 2)

    if size < 256:
        norm_square_img[extend: extend + size, extend: extend + size, :] = squared_img
        norm_square_mask[extend: extend + size, extend: extend + size, :] = squared_mask

        return norm_square_img, norm_square_mask

    return squared_img.astype(float16), squared_mask


def split_component(image, labeled_mask, bbox, loop=0):
    center_w = int((bbox[1] + bbox[0]) / 2)
    delta = int((bbox[3] - bbox[2]) * .4)
    tmp_1 = labeled_mask[bbox[2] + delta: bbox[3] - delta, bbox[0] - 20:center_w + 20]

    tmp_2 = labeled_mask[bbox[2] + delta: bbox[3] - delta, center_w - 20: bbox[1] + 20]

    if loop == 0:

        img1 = image[bbox[2] + delta: bbox[3] - delta, bbox[0] - 20:center_w + 20]
        img1 = (img1 - np.amin(img1)) * 255 / (np.amax(img1) - np.amin(img1))

        gray_image1 = np.uint8(img1)

        _, binary_img1 = cv2.threshold(gray_image1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img2 = image[bbox[2] + delta: bbox[3] - delta, center_w - 20: bbox[1] + 20]

        img2 = (img2 - np.amin(img2)) * 255 / (np.amax(img2) - np.amin(img2))

        gray_image2 = np.uint8(img2)

        _, binary_img2 = cv2.threshold(gray_image2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        labeled_mask[bbox[2] + delta: bbox[3] - delta, bbox[0] - 20:center_w + 20] = np.logical_and(tmp_1,
                                                                                                    np.logical_not(
                                                                                                        binary_img1))
        labeled_mask[bbox[2] + delta: bbox[3] - delta, center_w - 20: bbox[1] + 20] = np.logical_and(tmp_2,
                                                                                                     np.logical_not(
                                                                                                         binary_img2))
    else:

        min_val = 1e6
        index = 0

        if np.sum(tmp_1) > 0:
            _bbox = bbox_2D(tmp_1)

            # print(_bbox)

            for i in range(_bbox[2] + 10, _bbox[3] - 10):
                if np.sum(tmp_1[i, :]) < min_val:
                    min_val = np.sum(tmp_1[i, :])
                    index = i
                if np.sum(tmp_1[i, :]) == min_val:
                    index = int((index + i) / 2)
            tmp_1[index, :] = 0

        if np.sum(tmp_2) > 0:
            _bbox = bbox_2D(tmp_2)

            min_val = 1e6
            index = 0
            for i in range(_bbox[2] + 10, _bbox[3] - 10):
                if np.sum(tmp_2[i, :]) < min_val:
                    min_val = np.sum(tmp_2[i, :])
                    index = i
                if np.sum(tmp_2[i, :]) == min_val:
                    index = int((index + i) / 2)
            tmp_2[index, :] = 0

    return labeled_mask


def split_mask(image, mask, local_box, loop=0):
    new_mask = np.zeros(mask.shape)
    center_y = int((local_box[0] + local_box[1]) / 2)

    if np.any(mask):
        labeled_mask, num_cc = label(mask)

        for i in range(1, num_cc + 1):
            label_mask = (labeled_mask == i).astype("uint8")
            area = np.sum(label_mask)
            upper_area = np.sum(label_mask[:center_y, :])
            lower_area = np.sum(label_mask[center_y:, :])

            bbox = bbox_2D(label_mask)
            height = bbox[3] - bbox[2]

            # print(area, upper_area, lower_area)

            if area < image.shape[0]:
                # print("Small")
                new_mask += label_mask * 10 if upper_area > lower_area else label_mask * 20

            elif upper_area / area > 0.8 and height < 256:
                # print("Up")
                new_mask += label_mask * 10

            elif lower_area / area > 0.8 and height < 256:
                # print("Low")
                new_mask += label_mask * 20

            else:
                _label_mask = split_component(image, label_mask, bbox, loop)

                new_mask += split_mask(image, _label_mask, local_box, loop + 1)

    return new_mask


def load_weights(model, segment_model_folder, segment_model_name):
    if os.path.exists(os.path.join(segment_model_folder, f"{segment_model_name}.index")):
        print(segment_model_name)
        model.load_weights(os.path.join(segment_model_folder, segment_model_name))

        return model

    return 0


def resize_3d_image_cubic(image):
    # Load the 3D image using nibabel

    # Get the original image dimensions
    original_shape = image.shape

    # Calculate the scaling factors for each dimension
    scale_factors = [1, 100 / 512, 448 / original_shape[2]]

    # Resize the 3D image using cubic interpolation
    resized_data = zoom(image, scale_factors, order=3)

    return resized_data


def norm_image(layer):
    filter_img = np.logical_and(-1500 < layer, layer < 1000)
    # filtered_img = np.multiply(img, filter_img)

    img = np.multiply(layer, filter_img) + np.logical_not(filter_img) * -1500
    filtered_image = img - np.amin(img)
    return (filtered_image / (np.amax(filtered_image) - np.amin(filtered_image))) * 255


def recreate_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Delete the directory and its contents
        shutil.rmtree(directory_path)

    # Create the new directory
    os.mkdir(directory_path)


def gaussian_weighting(image, sigma=1.0, threshold=0.5, min_region_size=100, voxel_ratio=None):
    # print(voxel_ratio)
    # Tìm các thành phần liên thông có giá trị > threshold
    labeled_image, num_labels = label(image > threshold)

    # Tính trọng tâm và khối lượng của từng vùng liên thông
    regions = regionprops(labeled_image)

    sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)
    # Chuẩn bị một mảng trọng số trống
    weighted_image = np.zeros_like(image, dtype=float)

    # Áp dụng trọng số Gaussian cho từng vùng liên thông
    for region in sorted_regions:
        if region.area >= min_region_size:
            center = region.centroid

            weights = compute_gaussian(image.shape, np.array(center).astype(int), voxel_ratio, sigma)

            weighted_image += weights
        else:
            break

    return weighted_image if np.amax(weighted_image) > 0 else np.ones_like(image, dtype=float)


def compute_gaussian(tile_size: Union[Tuple[int, ...], List[int]], center_coords, voxel_ratio,
                     sigma_scale: float = 1. / 8
                     , value_scaling_factor: float = 1):
    tmp = np.zeros(tile_size)
    sigmas = [tile_size[0] * sigma_scale] * 3
    sigmas[-1] = sigmas[-1] / voxel_ratio
    tmp[center_coords] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    gaussian_importance_map = gaussian_importance_map / np.amax(gaussian_importance_map) * value_scaling_factor

    gaussian_importance_map[gaussian_importance_map == 0] = np.amin(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map
