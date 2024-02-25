import numpy as np

from utils import bbox_location, find_and_fill_contours


def prepare_mask(mask):
    new_mask = np.zeros(mask.shape)
    bbox = bbox_location(mask)

    for i in range(bbox[4], bbox[5] + 1):
        new_mask[:, :, i] = find_and_fill_contours(mask[:, :, i])

    return new_mask


def intensity_normalize(image, max_value, min_value, mean_intensity, std_intensity, ggo_mask=None):
    """
    Normalize voxel value from 0 to 1 with range from 0.5 percent to 99.5 percent in foreground image
    :return:
    """
    _mask = np.logical_and(image >= min_value, image <= max_value)

    image = np.clip(image, min_value, max_value)
    image = (image - mean_intensity) / max(std_intensity, 1e-8)

    if ggo_mask is None:
        return image

    return image, np.logical_and(ggo_mask, _mask)


def preprocess(image, max_value, min_value, mean_intensity, std_intensity,
               ggo_mask=None, lung_mask=None):
    img, ggo = intensity_normalize(image, max_value, min_value, mean_intensity, std_intensity, ggo_mask)

    if ggo_mask is not None:
        print("Fill mask")
        ggo_mask = prepare_mask(ggo)

    return img, ggo_mask, lung_mask
