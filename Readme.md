- Cài đặt thư viện cần thiết:

pip install -r requirements.txt

- Huấn luyện mô hình phân đoạn phổi
Chuẩn bị dữ liệu: 


cd ggos_segment/dataset/
tfds new lung_dataset

Chuẩn bị bộ dữ liệu phân đoạn phổi sau đó build:

tfds build

Huấn luyện mô hình phân đoạn phổi:

cd ggos_segment
python train_lung_mask.py

Phân đoạn và hậu xử lý mặt nạ phổi

python ./prediction/lung_prediction.py
python lung_post_process.py

Huấn luyện mô hình phân đoạn GGOs

Chuẩn bị dữ liệu:

cd prepare_data
python generate_2d_data.py
python generate_3d_data.py

cd ../ggos_segment/dataset/

cd lggo_dataset
tfds build

cd sggo_dataset
tfds build

Huấn luyện mô hình 2D, 3D 

cd ggos_segment
python train_segment.py -f -b -d -l

-f : Fold
-b: batch_size
-d: 2 hoặc 3 (2D hoặc 3D)
-l: 0, 1, hoặc 2 (Loss id: 0: comboloss, 1: DiceFocal Loss, 2: Dice Boundary loss)

Phân đoạn dữ liệu 2D:

python ./prediction/ggo_prediction.py


Phân đoạn dữ liệu 3D:

python ./prediction/lggo_prediction.py

Hậu xử lý dữ liệu:

python post_process_ggo.py

Web:

cd ggo_segment_web
python app.py

Truy cập trang web tại http://127.0.0.1:5000/