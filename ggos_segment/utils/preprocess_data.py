import numpy as np
from scipy.ndimage import zoom

from utils.utils import bbox_location, find_and_fill_contours


def prepare_mask(mask):
    new_mask = np.zeros(mask.shape)
    bbox = bbox_location(mask)

    for i in range(bbox[4], bbox[5] + 1):
        new_mask[:, :, i] = find_and_fill_contours(mask[:, :, i])

    return new_mask


def crop_lung_mask(image, ggo_mask, lung_mask):
    """
    Crop CT image and GGO mask with bbox fit to lung mask
    :return:
    """
    bbox = bbox_location(lung_mask)
    return image[bbox[2]:bbox[3], bbox[0]:bbox[1], bbox[4]:bbox[5]], \
           ggo_mask[bbox[2]:bbox[3], bbox[0]:bbox[1], bbox[4]:bbox[5]], \
           lung_mask[bbox[2]:bbox[3], bbox[0]:bbox[1], bbox[4]:bbox[5]]


def resampling(image, current_spacing, target_spacing, lowres=False, ggo_mask=None, lung_mask=None):
    """
    Resampling image and mask to median voxel spacing
    :return:
    """
    # if target_spacing[-1] / current_spacing[-1] < 3:
    return [image], current_spacing, [ggo_mask], [lung_mask], [image.shape]

    # bbox = bbox_location(ggo_mask)
    #
    # if bbox[5] - bbox[4] < 40:
    #     return [image], current_spacing, [ggo_mask], [lung_mask]

    # img = []
    # segs = []
    # lungs = []
    # org_shapes = []
    #
    # if lowres:
    #     spacing = [current_spacing[0], current_spacing[1], current_spacing[2] * 3]
    #     target_spacing = [target_spacing[0], target_spacing[1], target_spacing[2]]
    #
    # else:
    #     spacing = [current_spacing[0], current_spacing[1], current_spacing[2] * 3]
    #     target_spacing = [current_spacing[0], current_spacing[1], target_spacing[2]]
    #
    # for i in range(3):
    #     im = resample_patch(image[:, :, i::3], spacing, target_spacing, False)
    #     img.append(im)
    #     org_shapes.append(image[:, :, i::3].shape)
    #     if ggo_mask is not None:
    #         seg = resample_patch(ggo_mask[:, :, i::3], spacing, target_spacing, True)
    #         segs.append(seg)
    #     if lung_mask is not None:
    #         lung = resample_patch(lung_mask[:, :, i::3], spacing, target_spacing, True)
    #         lungs.append(lung)
    #
    # return img, target_spacing, segs, lungs, org_shapes


def resample_patch(patch, old_spacing, new_spacing, is_seg=False, to_original = False):
    # Compute the resampling factors for each dimension
    new_xy_spacing = [new_spacing[0], new_spacing[1], old_spacing[2]]
    resampling_factors_x_y = np.divide(old_spacing, new_xy_spacing)
    resampling_factors_z = np.divide(new_xy_spacing, new_spacing)

    # print(resampling_factors_z)

    # Resample the patch using trilinear interpolation
    if is_seg:
        resampled_patch = zoom(patch, resampling_factors_x_y, order=1)
    else:
        resampled_patch = zoom(patch, resampling_factors_x_y, order=3)

    if not to_original:
        resampled_patch = zoom(resampled_patch, resampling_factors_z, order=0)
    else:
        resampled_patch = zoom(resampled_patch, resampling_factors_z, order=1)

    return resampled_patch


def intensity_normalize(image, max_value, min_value, mean_intensity, std_intensity):
    """
    Normalize voxel value from 0 to 1 with range from 0.5 percent to 99.5 percent in foreground image
    :return:
    """

    image = np.clip(image, min_value, max_value)
    image = (image - mean_intensity) / max(std_intensity, 1e-8)

    return image


def preprocess(image, current_spacing, target_spacing, max_value, min_value, mean_intensity, std_intensity,
               ggo_mask=None, lung_mask=None, lowres=False):
    # if ggo_mask is not None:
    #     print("Fill mask")
    #     ggo_mask = prepare_mask(ggo_mask)

    # print("Intensity norm")
    img = intensity_normalize(image, max_value, min_value, mean_intensity, std_intensity)

    if ggo_mask is not None:
        print("Fill mask")
        ggo_mask = prepare_mask(ggo_mask)

    print("Resampling")
    ims, spacing, ggos, lungs, org_shapes = resampling(img, current_spacing, target_spacing, lowres, ggo_mask,
                                                       lung_mask)
    # normalized_images = []
    # for i in range(len(ims)):
    #     normalized_images.append(intensity_normalize(ims[i], max_value, min_value, mean_intensity, std_intensity))

    return ims, ggos, lungs, spacing, org_shapes


def return_original(masks, org_shapes):
    if len(masks) == 1:
        shape = masks[0].shape
        if org_shapes[0][0] == shape[0] and org_shapes[0][2] == shape[2]:
            return masks[0]

        else:
            return resample_patch(masks[0], shape, org_shapes[0], True)

    org_masks = []

    z_size = masks[0].shape[2] + masks[1].shape[2] + masks[2].shape[2]

    cur_shape = (org_shapes[0][0], org_shapes[0][1], z_size)

    result = np.zeros((org_shapes[0][0], org_shapes[0][1], z_size))

    # print(np.sum(masks[0])).

    for i in range(3):
        result[:, :, i::3] = masks[i]

    # for mask, org_shape in zip(masks, org_shapes):
    #     shape = mask.shape
    #     # print(shape)
    #     if org_shape[0] == shape[0] and org_shape[2] == shape[2]:
    #         org_masks.append(mask)
    #
    #     else:
    #         org_masks.append(resample_patch(mask, org_shape, shape, True, True))

    # print(np.sum(org_masks[0]))

    org_shape = (org_shapes[0][0], org_shapes[0][1], org_shapes[0][2] + org_shapes[1][2] + org_shapes[2][2])

    return resample_patch(result, org_shape, cur_shape, True, True)
