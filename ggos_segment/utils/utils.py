from typing import Union, Tuple, List

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import label, gaussian_filter

from scipy.ndimage import label, gaussian_filter
from skimage.measure import regionprops


def gaussian_weighting(image, sigma=1.0, threshold=0.5, min_region_size=100, voxel_ratio=None):
    # Tìm các thành phần liên thông có giá trị > threshold
    labeled_image, num_labels = label(image > threshold)

    # Tính trọng tâm và khối lượng của từng vùng liên thông
    regions = regionprops(labeled_image)

    # Chuẩn bị một mảng trọng số trống
    weighted_image = np.zeros_like(image, dtype=float)

    # Áp dụng trọng số Gaussian cho từng vùng liên thông
    for region in regions:
        if region.area >= min_region_size:
            center = region.centroid

            weights = compute_gaussian(image.shape, np.array(center).astype(int), voxel_ratio, sigma)

            weighted_image += weights

    return weighted_image if np.amax(weighted_image) > 0 else np.ones_like(image, dtype=float)


def count_connected_components_3d(mask):
    # label the connected components in the mask
    labeled_mask, num_cc = label(mask)

    return num_cc


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
            img = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            mask[:, :, i] = img

    return (mask > 0).astype(int)


def bbox_2D(mask):
    image1_non_zero = np.nonzero(mask)
    x_min = np.amin(image1_non_zero[1])
    x_max = np.amax(image1_non_zero[1])

    y_min = np.amin(image1_non_zero[0])
    y_max = np.amax(image1_non_zero[0])

    return x_min, x_max, y_min, y_max, (x_max - x_min) * (y_max - y_min)


def erode_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for i in range(mask.shape[-1]):
        img = mask[:, :, i]
        plt.imsave("tmp.png", img, cmap="gray")
        img = cv2.imread("tmp.png", 0)
        if np.sum(img) > 10:
            # img = cv2.GaussianBlur(img, (3, 3), 0)
            img = cv2.erode(img, kernel)
            mask[:, :, i] = img

    return (mask > 0).astype(int)


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

    norm_square_img = np.ones((224, 224)) * (min_value - 1)
    norm_square_mask = np.zeros((224, 224))
    extend = int((224 - size) / 2)

    if size < 224:
        norm_square_img[extend: extend + size, extend: extend + size] = squared_img
        norm_square_mask[extend: extend + size, extend: extend + size] = squared_mask

        return norm_square_img, norm_square_mask

    return squared_img, squared_mask


def square_3D_image_and_mask(img, mask, min_value):
    size = np.amax(img.shape[:2])
    dif = int((np.amax(img.shape[:2]) - np.amin(img.shape[:2])) / 2)
    squared_img = np.ones((size, size, 10)) * (min_value - 1)
    squared_mask = np.zeros((size, size, 10))

    if img.shape[0] > img.shape[1]:
        squared_img[:, dif: dif + img.shape[1], :] = img
        squared_mask[:, dif: dif + img.shape[1], :] = mask

    else:
        squared_img[dif: dif + img.shape[0], :, :] = img
        squared_mask[dif: dif + img.shape[0], :, :] = mask

    norm_square_img = np.ones((224, 224, 10)) * (min_value - 1)
    norm_square_mask = np.zeros((224, 224, 10))
    extend = int((224 - size) / 2)

    if size < 224:
        norm_square_img[extend: extend + size, extend: extend + size, :] = squared_img
        norm_square_mask[extend: extend + size, extend: extend + size, :] = squared_mask

        return norm_square_img, norm_square_mask

    return squared_img, squared_mask


def split_component(image, labeled_mask, bbox):
    center = int((bbox[2] + bbox[3]) / 2)
    delta = int((bbox[3] - bbox[2]) * 0.2)
    tmp = labeled_mask[center - delta: center + delta, :].copy()

    _labeled_mask, num_cc = label(tmp)

    if not np.any(tmp):
        plt.imshow(labeled_mask)
        plt.show()

    tmp_bbox = bbox_2D(tmp)

    for i in range(1, num_cc + 1):
        _label = (_labeled_mask == i).astype(int)
        _bbox = bbox_2D(_label)
        if _bbox[0] == 0 and _bbox[1] == tmp.shape[1]:
            tmp_bbox = _bbox

        else:
            continue

    tmp = labeled_mask[center - delta: center + delta, tmp_bbox[0] - 10:tmp_bbox[1] + 10]

    min_val = 1e6
    index = 0
    for i in range(tmp.shape[0]):
        if np.sum(tmp[i, :]) < min_val:
            min_val = np.sum(tmp[i, :])
            index = i
        if np.sum(tmp[i, :]) == min_val:
            index = int((index + i) / 2)

    tmp[index, :] = 0

    return labeled_mask


def split_mask(image, mask, local_box):
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

            if area < 500:
                # print("Small")
                new_mask += label_mask * 10 if upper_area > lower_area else label_mask * 20

            elif upper_area / area > 0.8:
                # print("Up")
                new_mask += label_mask * 10

            elif lower_area / area > 0.8:
                # print("Low")
                new_mask += label_mask * 20

            else:
                # print("Oke")
                _label_mask = split_component(image, label_mask, bbox)

                # plt.imshow(_label_mask)
                # plt.show()

                new_mask += split_mask(image, _label_mask, local_box)

    return new_mask


def compute_gaussian(tile_size: Union[Tuple[int, ...], List[int]], center_coords, voxel_ratio,
                     sigma_scale: float = 1. / 8
                     , value_scaling_factor: float = 1):
    tmp = np.zeros(tile_size)
    # center_coords = [i // 2 for i in tile_size]
    # print(center_coords)
    sigmas = [tile_size[0] * sigma_scale] * 3
    sigmas[-1] = sigmas[-1] / voxel_ratio
    tmp[center_coords] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    # gaussian_importance_map = torch.from_numpy(gaussian_importance_map).type(dtype).to(device)

    gaussian_importance_map = gaussian_importance_map / np.amax(gaussian_importance_map) * value_scaling_factor
    # gaussian_importance_map = gaussian_importance_map.type(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.amin(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map
