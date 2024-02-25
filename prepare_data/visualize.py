import os.path

import numpy as np
from matplotlib import pyplot as plt
from skimage.exposure import exposure


def equalize_histogram_3d(image):
    # get the shape of the image
    z = image.shape[-1]

    img = (image - np.amin(image)) / (np.amax(image) - np.amin(image))

    # create a new array to store the equalized image
    equalized_image = np.zeros_like(image)

    # loop through each layer of the image in the Z axis
    for i in range(z):
        # get the current layer
        layer = img[:, :, i]

        # equalize the histogram of the layer
        equalized_layer = exposure.equalize_hist(layer)

        # add the equalized layer to the new image array
        equalized_image[:, :, i] = equalized_layer

    return equalized_image * (np.amax(image) - np.amin(image)) + np.amin(image)


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def multi_slice_viewer(volume, mask):
    remove_keymap_conflicts({'left', 'right'})  # Modify key conflicts
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.volume = volume
    fig.mask = mask
    ax_img, ax_mask = axs[0], axs[1]
    ax_img.volume = volume
    ax_img.index = volume.shape[2] // 2  # Starting index along the z-axis
    ax_img.imshow(volume[:, :, ax_img.index], cmap="gray")
    ax_img.set_title('Image')
    ax_mask.volume = mask
    ax_mask.index = volume.shape[2] // 2  # Starting index along the z-axis
    ax_mask.imshow(mask[:, :, ax_mask.index], cmap="gray")
    ax_mask.set_title('Mask')
    fig.canvas.mpl_connect('key_press_event', process_key)


def process_key(event):
    fig = event.canvas.figure
    ax_img, ax_mask = fig.axes[0], fig.axes[1]
    if event.key == 'left':  # Change key condition
        previous_slice(ax_img, ax_mask)
    elif event.key == 'right':  # Change key condition
        next_slice(ax_img, ax_mask)
    fig.canvas.draw()


def previous_slice(ax_img, ax_mask):
    volume_img = ax_img.volume
    volume_mask = ax_mask.volume
    ax_img.index = (ax_img.index - 1) % volume_img.shape[2]  # Update index along the z-axis
    ax_mask.index = (ax_mask.index - 1) % volume_mask.shape[2]  # Update index along the z-axis
    ax_img.images[0].set_array(volume_img[:, :, ax_img.index])
    ax_mask.images[0].set_array(volume_mask[:, :, ax_mask.index])


def next_slice(ax_img, ax_mask):
    volume_img = ax_img.volume
    volume_mask = ax_mask.volume
    ax_img.index = (ax_img.index + 1) % volume_img.shape[2]  # Update index along the z-axis
    ax_mask.index = (ax_mask.index + 1) % volume_mask.shape[2]  # Update index along the z-axis
    ax_img.images[0].set_array(volume_img[:, :, ax_img.index])
    ax_mask.images[0].set_array(volume_mask[:, :, ax_mask.index])


def show_ggo_slide(image, mask, pred_mask):
    z = image.shape[-1]

    for i in range(z):
        if np.any(mask[:, :, i]) or np.any(pred_mask[:, :, i]):
            fig, axs = plt.subplots(1, 3, figsize=(16, 8))
            axs[0].imshow(image[:, :, i])
            axs[1].imshow(mask[:, :, i])
            axs[2].imshow(pred_mask[:, :, i])

            plt.show()


train_dir = "Train"
import nibabel as nib

if __name__ == "__main__":
    for file in os.listdir(train_dir):
        # print(file)
        if file.endswith("_ct.nii.gz"):
            name = file[:len(file) - len("_ct.nii.gz")]
            im = nib.load(os.path.join(train_dir, file)).get_fdata()
            seg = nib.load(os.path.join(train_dir, f"{name}_seg.nii.gz")).get_fdata()

            multi_slice_viewer(im, seg)
            plt.show()
