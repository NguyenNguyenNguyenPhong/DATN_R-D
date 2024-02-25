import os
import argparse

import tensorflow_datasets as tfds
import tensorflow as tf
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from call_back import MyModelStorage
from metric.metric import precision, recall, f1_score
from loss.losses import dice_loss, weighted_binary_cross_entropy_loss, combo_loss, dice_focal_loss, dice_surface_loss
from arch.lggo_segment import LGGO_Segment, SGGO_Segment
from tensorflow.keras import mixed_precision
import numpy as np
import random
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter

model = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train segment model with specified fold and loss_id")
    parser.add_argument("-f", dest="fold", type=int, help="Fold value")
    parser.add_argument("-b", dest="batch", type=int, help="Batch size")
    parser.add_argument("-d", dest="dimension", type=int, help="Dimension")
    parser.add_argument("-lid", dest="loss_id", type=int, help="Loss ID")
    parser.add_argument("--continue", dest="continue_training", action="store_true",
                        help="Continue training from a checkpoint")
    return parser.parse_args()


def random_flip_data(image, mask):
    # print("Flip")
    """Randomly flip the image and mask along any axis."""
    mode = ["horizontal", "vertical", "horizontal_and_vertical"]
    axes = random.randint(0, 2)
    flipped_image = tf.keras.layers.RandomFlip(mode[axes])(tf.concat((image, mask), axis=-1))

    flipped_image = tf.cast(flipped_image, tf.float16)
    return flipped_image[:, :, 0:12], flipped_image[:, :, 12:]


def random_rotate_data(image, mask, rotation_range=(-10, 10)):
    """Randomly rotate the image and mask by an angle within the specified range."""

    rotated_image = tf.keras.layers.RandomRotation(
        (rotation_range[0] / 180 * 2 * np.pi, rotation_range[1] / 180 * 2 * np.pi))(tf.concat((image, mask), axis=-1))

    rotated_image = tf.cast(rotated_image, tf.float16)

    return rotated_image[:, :, 0:12], rotated_image[:, :, 12:]


def add_gaussian_noise(image, std_dev):
    """Add Gaussian noise to the image."""
    noise = tf.cast(tf.keras.layers.GaussianNoise(std_dev)(image), tf.float16)
    noisy_image = image + noise
    return noisy_image


def load_image(datapoint):
    input_image = tf.expand_dims(datapoint['images'], axis=-1)
    label = tf.expand_dims(datapoint['label'], axis=-1)

    # input_image = add_gaussian_noise(input_image, 0.1)
    # input_image = color_augmentation(input_image)
    # input_image, label = random_flip_data(input_image, label)

    return input_image, label


def train_segment_model():
    global model
    global sample_image
    global sample_mask

    args = parse_args()
    test_fold = args.fold
    loss_id = args.loss_id
    continue_training = args.continue_training
    dim = args.dimension

    segment_model_folder = "weight/ggo"
    data = []

    if dim == 3:
        lggo = LGGO_Segment((256, 256, 12, 1), channel=32, stage=5)
        model = lggo.unet
        dataset_name = "large_ggo_dataset"
        file_name = "LGGO"

    else:
        sggo = SGGO_Segment((256, 256, 1), channel=32, stage=6)
        model = sggo.unet
        dataset_name = "ggo_dataset"
        file_name = "SGGO"

    BATCH_SIZE = args.batch
    BUFFER_SIZE = 1000
    EPOCHS = 100

    if loss_id == 0:
        loss_name = "CBL"
        loss_func = combo_loss
    elif loss_id == 1:
        loss_name = "DFL"
        loss_func = dice_focal_loss
    else:
        loss_name = "DBL"
        loss_func = dice_surface_loss

    new_segment_model_name = f"{file_name}_segment_32_5_{loss_name}_fold_{test_fold}"

    dataset, info = tfds.load(dataset_name, with_info=True)

    for i in range(5):
        fold = dataset[f"fold_{i}"].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        data.append(fold)

    train_images = data[0] if test_fold != 0 else data[1]

    for i in range(1, 5):

        if test_fold == 0 and i == 1:
            continue
        if i != test_fold:
            train_images = train_images.concatenate(data[i])

    test_images = data[test_fold]

    VAL_SUBSPLITS = 5
    STEPS_PER_EPOCH = train_images.cardinality().numpy() // BATCH_SIZE
    VALIDATION_STEPS = info.splits[f'fold_{test_fold}'].num_examples // BATCH_SIZE // VAL_SUBSPLITS

    train_batches = (
        train_images
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .cache()
            .repeat()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
    )
    test_batches = test_images.batch(BATCH_SIZE)

    mixed_precision.set_global_policy('mixed_float16')

    model.summary()

    if continue_training:
        load_weights(model, segment_model_folder, new_segment_model_name)

    lr = 1e-4

    model.compile(optimizer=Adam(lr),
                  loss=loss_func,
                  metrics=["accuracy", f1_score, precision, recall])

    model_history = model.fit(train_batches,
                              epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_batches,
                              callbacks=[MyModelStorage(model, segment_model_folder, new_segment_model_name),
                                         ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-7,
                                                           verbose=1),
                                         ])


if __name__ == '__main__':
    train_segment_model()
