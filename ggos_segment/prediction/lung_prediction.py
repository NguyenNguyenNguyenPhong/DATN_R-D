import os
import pathlib
import cv2
import numpy as np
from matplotlib import pyplot as plt
import nibabel as nib
from arch.lung_model import build_unet
import tensorflow as tf

segment_model_folder = "D:\\Github\\lungs_segmentation\\train\\weight\\ggo"

lung_segment_model_name = "lung_segment"

dataset_folder = pathlib.Path(__file__).parent.parent.resolve()
print(dataset_folder)

image_dir = os.path.join("D://Github", "data/COVID-19-20/COVID-19-20_v2/test")
pred_mask_dir = "../data/dataset/new_lung_test_mask"

model = build_unet(input_shape=(224, 224, 3), chanel=32, stage=5)
model.load_weights(os.path.join(segment_model_folder, lung_segment_model_name))


def norm_image(layer):
    filter_img = np.logical_and(-1500 < layer, layer < 1000)
    # filtered_img = np.multiply(img, filter_img)

    img = np.multiply(layer, filter_img) + np.logical_not(filter_img) * -1500
    filtered_image = img - np.amin(img)
    return (filtered_image / (np.amax(filtered_image) - np.amin(filtered_image))) * 255


def predict(layer):
    layer = norm_image(layer)
    shape = layer.shape
    plt.imsave("tmp_img.png", layer, cmap='gray')
    img = cv2.imread("tmp_img.png")
    img = tf.image.resize(img, (224, 224))

    # img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    mask = model.predict(img[tf.newaxis, ...])
    # mask = mask.astype(float)
    mask = np.where(mask > 0.5, 1, 0).astype(int)

    mask = tf.image.resize(mask, (shape[0], shape[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.squeeze(mask, -1)

    return mask[0]


def predict_mask(image):
    z = image.shape[-1]

    # initialize a list to store the number of connected components in each layer
    mask = np.zeros(image.shape)

    # loop through each layer of the bitmaps in the Z axis
    for i in range(z):
        layer = image[:, :, i]

        pred_i = predict(layer)

        mask[:, :, i] = pred_i

    return mask

check = False
for file in os.listdir(image_dir):

    if file.endswith(".nii.gz"):

        file_name = file.split(".")[0]

        if file_name == "radiopaedia_10_85902_1":
            check = True

        if not check:
            continue

        ct_load = nib.load(os.path.join(image_dir, file))

        print(file_name)

        array_image = ct_load.get_fdata()

        pred_mask = predict_mask(array_image)

        seg = nib.Nifti1Image(pred_mask, ct_load.affine, ct_load.header)

        pred_mask_name = os.path.join(pred_mask_dir, file_name + ".nii.gz")
        nib.save(seg, pred_mask_name)
