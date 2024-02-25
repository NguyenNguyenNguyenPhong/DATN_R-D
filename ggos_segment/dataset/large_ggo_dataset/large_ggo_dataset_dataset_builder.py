"""large_ggo_dataset dataset."""

import os
import pathlib

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from numpy import int16

dataset_folder = pathlib.Path(__file__).parent.resolve()

# print(dataset_folder)
annotations_folder = 'D:/Github/data/COVID-19-20/COVID-19-20_v2/3d_segment_data'
# print(annotations_folder)

image_folder = os.path.join("D:/Github/data/COVID-19-20/COVID-19-20_v2/3d_segment_data/image")


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for lung_dataset images."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the images metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict({
                'images': tfds.features.Tensor(shape=(256, 256, 12), dtype=tf.float16, encoding='bytes'),
                'label': tfds.features.Tensor(shape=(256, 256, 12), dtype=tf.float16, encoding='bytes'),
            }),
            supervised_keys=('images', 'label'),
            homepage='https://dataset-homepage/',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        fold_0 = tfds.core.SplitGenerator(
            name="fold_0",
            gen_kwargs={
                "images_dir_path":
                    image_folder,
                "annotations_dir_path":
                    annotations_folder,
                "images_list_file":
                    os.path.join(annotations_folder, "fold_0.txt")
            }
        )

        fold_1 = tfds.core.SplitGenerator(
            name="fold_1",
            gen_kwargs={
                "images_dir_path":
                    image_folder,
                "annotations_dir_path":
                    annotations_folder,
                "images_list_file":
                    os.path.join(annotations_folder, "fold_1.txt")
            }
        )

        fold_2 = tfds.core.SplitGenerator(
            name="fold_2",
            gen_kwargs={
                "images_dir_path":
                    image_folder,
                "annotations_dir_path":
                    annotations_folder,
                "images_list_file":
                    os.path.join(annotations_folder, "fold_2.txt")
            }
        )

        fold_3 = tfds.core.SplitGenerator(
            name="fold_3",
            gen_kwargs={
                "images_dir_path":
                    image_folder,
                "annotations_dir_path":
                    annotations_folder,
                "images_list_file":
                    os.path.join(annotations_folder, "fold_3.txt")
            }
        )

        fold_4 = tfds.core.SplitGenerator(
            name="fold_4",
            gen_kwargs={
                "images_dir_path":
                    image_folder,
                "annotations_dir_path":
                    annotations_folder,
                "images_list_file":
                    os.path.join(annotations_folder, "fold_4.txt")
            }
        )

        return [fold_0, fold_1, fold_2, fold_3, fold_4]

    def _generate_examples(self, images_dir_path, annotations_dir_path, images_list_file):
        with tf.io.gfile.GFile(images_list_file, "r") as images_list:
            for line in images_list:
                image_name = line.strip()
                bitmaps_dir_path = os.path.join(annotations_dir_path, "mask")

                bitmap_name = image_name + ".npy"
                image_name += ".npy"

                image_path = os.path.join(images_dir_path, image_name)
                images_data = np.load(image_path)
                # images_data = images_data.astype(int16)

                label_path = os.path.join(bitmaps_dir_path, bitmap_name)
                label_data = np.load(label_path)

                record = {
                    "images": images_data,
                    "label": label_data,
                }
                yield image_name, record
