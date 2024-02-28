import os
import shutil
import threading
import time
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, jsonify
from scipy.ndimage import zoom

from config import UPLOAD_FOLDER, SEGMENT_FOLDER
from segmentation import do_segmentation
from utils import resize_3d_image_cubic, norm_image, recreate_directory

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def plot_3d_image(image_path, filename):
    # Đọc dữ liệu hình ảnh nii.
    img = nib.load(image_path)
    data = img.get_fdata()

    voxel_spacing = img.header.get_zooms()
    voxel_ratio = voxel_spacing[-1] / voxel_spacing[0]

    app.config['VOXEL_RATIO'] = voxel_ratio

    name = filename.split(".")[0]

    resize_data = resize_3d_image_cubic(data)

    trans_data = np.transpose(resize_data, (2, 0, 1))

    # Tạo mảng 3D từ dữ liệu hình ảnh.
    slices_eval = [data[:, :, i] for i in range(data.shape[2])]
    slices_site = [trans_data[:, :, i] for i in range(trans_data.shape[2])]

    # Tạo các hình ảnh từ mảng 3D và lưu vào thư mục.
    output_side_folder = os.path.join(UPLOAD_FOLDER, "image_side")
    output_elevation_folder = os.path.join(UPLOAD_FOLDER, "image_elevation")
    recreate_directory(output_side_folder)
    recreate_directory(output_elevation_folder)

    for i, slice_data in enumerate(slices_eval):
        img = norm_image(slice_data)
        plt.imsave(os.path.join(output_elevation_folder, f"{name}_{i}.png"), img, cmap="gray")

    for i, slice_data in enumerate(slices_site):
        img_array = np.flip(slice_data, 0)
        img = norm_image(img_array)
        plt.imsave(os.path.join(output_side_folder, f"{name}_{i}.png"), img, cmap="gray")

    return len(slices_eval), len(slices_site)


def run_segmentation(image_path, name):
    print("segment")
    img = nib.load(image_path)
    data = img.get_fdata()
    segment_name, num_img, num_trans = do_segmentation(data, name, app.config['VOXEL_RATIO'],
                                                       app.config['USE_ENSEMBLE'])
    # This function will run the AI process in the background
    # segmented_filename = fake_perform_segmentation(image_path)
    enable_segment_button(segment_name, num_img, num_trans)


def enable_segment_button(segmented_filename, num_1, num_2):
    # This function will be called after the AI process is completed
    with app.app_context():
        app.config['SEGMENTED_FILENAME'] = segmented_filename
        app.config['NUM_ELEVATION'] = num_1
        app.config['NUM_SIDE'] = num_2


@app.route('/get_segmented_filename')
def get_segmented_filename():
    segmented_filename = app.config['SEGMENTED_FILENAME']
    num_ele = app.config['NUM_ELEVATION']
    num_side = app.config['NUM_SIDE']
    return jsonify({'segmented_filename': segmented_filename, 'num_ele': num_ele, 'num_side': num_side})


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    app.config['SEGMENTED_FILENAME'] = None
    app.config['NUM_ELEVATION'] = None
    app.config['NUM_SIDE'] = None
    app.config['VOXEL_RATIO'] = None
    app.config['USE_ENSEMBLE'] = False
    if request.method == 'POST':
        file = request.files['image']
        use_ensemble = request.form.get('use_ensemble')

        app.config['USE_ENSEMBLE'] = use_ensemble

        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if use_ensemble:
                # Thực hiện phân đoạn sử dụng Ensemble (nếu cần).
                pass

            return redirect(url_for('display_image', filename=filename))

    return render_template('index.html')


@app.route('/display/<filename>')
def display_image(filename):
    print(filename)
    name = filename.split(".")[0]
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    num_slices_eval, num_slices_site = plot_3d_image(image_path, filename)
    uploaded_image_paths_1 = [
        url_for('static', filename=f'upload/image_elevation/{name}_{i}.png') for i in range(num_slices_eval)
    ]

    uploaded_image_paths_2 = [
        url_for('static', filename=f'upload/image_side/{name}_{i}.png') for i in range(num_slices_site)
    ]

    segmentation_thread = threading.Thread(target=run_segmentation, args=(image_path, name))
    segmentation_thread.start()

    return render_template('display.html', name=name, uploaded_image_paths_1=uploaded_image_paths_1,
                           uploaded_image_paths_2=uploaded_image_paths_2,
                           time=int(num_slices_eval * 1.5) if app.config['USE_ENSEMBLE'] == True else num_slices_eval)
    # return render_template('display.html')


@app.route('/segmentation_display/<filename>/<int:num_ele>/<int:num_side>')
def display_segmentation(filename, num_ele, num_side):
    name = filename.split(".")[0]

    uploaded_image_paths_1 = [
        url_for('static', filename=f'lung_segment/image_elevation/{name}_{i}.png') for i in range(num_ele)
    ]

    uploaded_image_paths_2 = [
        url_for('static', filename=f'lung_segment/image_side/{name}_{i}.png') for i in range(num_side)
    ]

    return render_template('segmentation.html', name=name, uploaded_image_paths_1=uploaded_image_paths_1,
                           uploaded_image_paths_2=uploaded_image_paths_2)


@app.route('/get_image_paths/<filename>')
def get_image_paths(filename):
    file_name = filename
    image_paths_1 = [url_for('static', filename=f'upload/image_elevation/{file_name}_{i}.png') for i in
                     range(len(os.listdir(os.path.join(UPLOAD_FOLDER, "image_elevation"))))]
    image_paths_2 = [url_for('static', filename=f'upload/image_side/{file_name}_{i}.png') for i in
                     range(len(os.listdir(os.path.join(UPLOAD_FOLDER, "image_side"))))]
    return jsonify({'image_paths_1': image_paths_1, 'image_paths_2': image_paths_2})


@app.route('/get_segment_image_paths/<filename>')
def get_segment_image_paths(filename):
    file_name = filename
    image_paths_1 = [url_for('static', filename=f'lung_segment/image_elevation/{file_name}_{i}.png') for i in
                     range(len(os.listdir(os.path.join(SEGMENT_FOLDER, "image_elevation"))))]
    image_paths_2 = [url_for('static', filename=f'lung_segment/image_side/{file_name}_{i}.png') for i in
                     range(len(os.listdir(os.path.join(SEGMENT_FOLDER, "image_side"))))]
    return jsonify({'image_paths_1': image_paths_1, 'image_paths_2': image_paths_2})


if __name__ == '__main__':
    app.run(debug=True)
