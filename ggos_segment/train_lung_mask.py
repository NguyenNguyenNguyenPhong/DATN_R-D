import os
import tensorflow_datasets as tfds
import tensorflow as tf
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from config import input_2d_shape, lr, segment_model_folder, LUNG_DATASET_NAME
from metric.metric import precision, recall, f1_score
from loss.losses import dice_loss, combo_loss, dice_focal_loss, surface_loss_keras, dice_surface_loss, \
    dice_focal_surface_loss
from arch.lggo_segment import build_unet

from call_back import MyModelStorage

segment_model_name = "lung_segment"


def load_image(datapoint):
    input_image = datapoint['images']

    label = datapoint['label']

    input_image = tf.image.resize(input_image, (224, 224))
    # input_image = tf.expand_dims(input_image, axis=-1)

    label = tf.image.resize(label, (224, 224), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # print(label)
    # label = tf.expand_dims(label, axis=-1)

    return input_image, label


def train_segment_model():
    BATCH_SIZE = 4
    BUFFER_SIZE = 1000
    EPOCHS = 50

    dataset_name = LUNG_DATASET_NAME

    dataset, info = tfds.load(dataset_name, with_info=True)

    train_images = dataset["train"].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    test_images = dataset["test"].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    VAL_SUBSPLITS = 5
    STEPS_PER_EPOCH = train_images.cardinality().numpy() // BATCH_SIZE
    VALIDATION_STEPS = test_images.cardinality().numpy() // BATCH_SIZE // VAL_SUBSPLITS
    train_batches = (
        train_images
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .repeat()
            .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    test_batches = test_images.batch(BATCH_SIZE)

    model = build_unet(input_shape=(224, 224, 3), chanel=32, stage=5)
    model.summary()
    # model.load_weights(os.path.join(segment_model_folder, segment_model_name))

    model.compile(optimizer=Adam(lr),
                  loss=dice_loss,
                  metrics=["accuracy", f1_score, precision, recall])

    model_history = model.fit(train_batches,
                              epochs=EPOCHS,
                              steps_per_epoch=STEPS_PER_EPOCH,
                              validation_steps=VALIDATION_STEPS,
                              validation_data=test_batches,
                              callbacks=[MyModelStorage(model, segment_model_name),
                                         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7,
                                                           verbose=1),
                                         ])

    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.figure()
    plt.plot(model_history.epoch, loss, 'r', label='Training loss')
    plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 0.1])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    train_segment_model()
