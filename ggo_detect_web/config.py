lung_model_name = "lung_segment"
sggo_model_name = "GGO_segment_DSFL_IN"
lggo_model_name = "LGGO_segment_32_5_IN_ATT_CBL"
sggo_best_fold = 2
lggo_best_fold = 1
model_path = "weight"

lung_input_size = (224, 224, 3)
sggo_input_size = (256, 256, 1)
lggo_input_size = (256, 256, 12, 1)

layer_3d_number = 12

UPLOAD_FOLDER = 'static/upload'
SEGMENT_FOLDER = "static/lung_segment"
voxel_max = 100.0
voxel_min = -1000.0


