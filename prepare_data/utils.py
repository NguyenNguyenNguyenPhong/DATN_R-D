import cv2
import numpy as np
from keras.backend import epsilon
from matplotlib import pyplot as plt
from numpy import int16, uint8, float16
from scipy.ndimage import label, gaussian_filter, binary_closing
from typing import Union, Tuple, List


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
    # if loop > 0:
    #     plt.imshow(mask)
    #     plt.show()
    new_mask = np.zeros(mask.shape)
    # print(local_box)
    center_y = int((local_box[0] + local_box[1]) / 2)

    print(center_y)

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


def dice_coefficient(y_true, y_pred):
    intersection = np.sum((y_true == 1) & (y_pred == 1))
    total = np.sum(y_true) + np.sum(y_pred)
    dice = (2.0 * intersection) / total
    return dice


def true_positive_rate(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives = np.sum(y_true == 1) + 1e-6
    tpr = tp / actual_positives
    return tpr


def true_negative_rate(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    actual_negatives = np.sum(y_true == 0) + 1e-6
    tnr = tn / actual_negatives
    return tnr


from scipy.ndimage import distance_transform_edt


def hausdorff_distance_3d(mask1, mask2, spacing, percentile=100):
    # Chuyển mask thành dạng boolean
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Tính khoảng cách Euclidean từ biên mask1 đến mask2 và ngược lại
    dist_1to2 = distance_transform_edt(~mask2, sampling=spacing)
    dist_2to1 = distance_transform_edt(~mask1, sampling=spacing)

    # Lấy danh sách các giá trị khoảng cách từ biên mask1 đến mask2 và từ biên mask2 đến mask1
    distances_mask1_to_mask2 = dist_1to2[mask1]
    distances_mask2_to_mask1 = dist_2to1[mask2]

    # Sắp xếp các giá trị khoảng cách tăng dần
    distances_mask1_to_mask2 = np.sort(distances_mask1_to_mask2)
    distances_mask2_to_mask1 = np.sort(distances_mask2_to_mask1)

    # Tính vị trí tương ứng với phần trăm cần tính (percentile)
    index_1to2 = int(np.ceil(percentile / 100 * len(distances_mask1_to_mask2))) - 1
    index_2to1 = int(np.ceil(percentile / 100 * len(distances_mask2_to_mask1))) - 1

    # Lấy giá trị khoảng cách tại vị trí tương ứng với phần trăm đã tính
    hausdorff_distance_1to2 = distances_mask1_to_mask2[index_1to2]
    hausdorff_distance_2to1 = distances_mask2_to_mask1[index_2to1]

    # Lấy Hausdorff distance lớn nhất
    hausdorff_distance = max(hausdorff_distance_1to2, hausdorff_distance_2to1)

    return hausdorff_distance


def surface_dice_3d(mask1, mask2):
    # Tính surface dice cho hai mask 3D

    # Chuyển mask thành dạng boolean
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Tính tổng số voxel trên biên giới hạn của từng mask
    surface_mask1 = mask1 ^ np.roll(mask1, 1, axis=0) ^ np.roll(mask1, -1, axis=0) ^ np.roll(mask1, 1,
                                                                                             axis=1) ^ np.roll(mask1,
                                                                                                               -1,
                                                                                                               axis=1) ^ np.roll(
        mask1, 1, axis=2) ^ np.roll(mask1, -1, axis=2)
    surface_mask2 = mask2 ^ np.roll(mask2, 1, axis=0) ^ np.roll(mask2, -1, axis=0) ^ np.roll(mask2, 1,
                                                                                             axis=1) ^ np.roll(mask2,
                                                                                                               -1,
                                                                                                               axis=1) ^ np.roll(
        mask2, 1, axis=2) ^ np.roll(mask2, -1, axis=2)

    # Tính số lượng voxel chung trên biên giới hạn của cả hai mask
    intersect_surface = surface_mask1 & surface_mask2

    # Tính Surface Dice
    surface_dice = 2 * intersect_surface.sum() / (surface_mask1.sum() + surface_mask2.sum())

    return surface_dice


def lggo_post_process(mask, voxel_ratio):
    selem = np.ones((3, 3, 3), dtype=np.bool)
    # closed_image = binary_closing(mask, structure=selem)

    # Smooth the closed image using a Gaussian filter with adjusted sigma
    sigma = 0.5  # Default sigma value
    if voxel_ratio is not None:
        # Adjust sigma based on voxel spacing ratio (voxel_ratio should be a tuple of (z, x, y) ratios)
        sigma = [sigma, sigma, sigma * voxel_ratio]

    smoothed_image = gaussian_filter(mask.astype(np.float), sigma=sigma)

    return (smoothed_image > 0.5).astype(int)