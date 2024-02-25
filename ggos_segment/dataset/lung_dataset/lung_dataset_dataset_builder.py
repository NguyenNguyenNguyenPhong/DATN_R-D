"""lung_dataset images."""
import os
import pathlib
import tensorflow as tf

import tensorflow_datasets as tfds

dataset_folder = pathlib.Path(__file__).parent.resolve()

# print(dataset_folder)
annotations_folder = os.path.join(dataset_folder, "annotations")
print(annotations_folder)

image_folder = os.path.join(dataset_folder, "images")


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for lung_dataset images."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the images metadata."""
        # TODO(lung_dataset): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # These are the features of your images like images, labels ...
                'images': tfds.features.Image(shape=(None, None, 3)),
                'label': tfds.features.Image(
                    shape=(None, None, 1), use_colormap=True
                ),
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('images', 'label'),  # Set to `None` to disable
            homepage='https://dataset-homepage/',
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        train_split = tfds.core.SplitGenerator(
            name="train",
            gen_kwargs={
                "images_dir_path":
                    image_folder,
                "annotations_dir_path":
                    annotations_folder,
                "images_list_file":
                    os.path.join(annotations_folder, "trainval.txt")
            }
        )

        test_split = tfds.core.SplitGenerator(
            name="test",
            gen_kwargs={
                "images_dir_path": image_folder,
                "annotations_dir_path": annotations_folder,
                "images_list_file": os.path.join(annotations_folder, "test.txt")
            },
        )

        return [train_split, test_split]

    def _generate_examples(self, images_dir_path, annotations_dir_path,
                           images_list_file):

        with tf.io.gfile.GFile(images_list_file, "r") as images_list:
            for line in images_list:
                image_name = line.strip()
                bitmaps_dir_path = os.path.join(annotations_dir_path, "bitmaps")

                bitmap_name = image_name + ".jpg"
                image_name += ".jpg"

                record = {
                    "images": os.path.join(images_dir_path, image_name),
                    "label": os.path.join(bitmaps_dir_path, bitmap_name),
                }
                yield image_name, record
