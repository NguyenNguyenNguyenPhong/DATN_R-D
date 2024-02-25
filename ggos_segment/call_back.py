import os
import tensorflow as tf

from config import segment_model_folder


class MyModelStorage(tf.keras.callbacks.Callback):
    def __init__(self,model,  model_path, model_name):
        super().__init__()
        self._min_method = False
        self._val_metric_name = "val_f1_score"
        self._curr_metric_val = None
        self._model_path = model_path
        self._model_name = model_name
        self._model = model

    def _log_value(self):
        print("\nmax metric_val " + str(self._curr_metric_val))

    def on_epoch_end(self, epoch, logs=None):
        metric_val = logs[self._val_metric_name]

        if self._curr_metric_val is None:
            self._curr_metric_val = metric_val
            return

        if self._min_method:
            if metric_val < self._curr_metric_val:
                print("\nSave_model metric_val " + str(metric_val) + " curr_metric_val " + str(self._curr_metric_val))
                self._curr_metric_val = metric_val

                save_model(self._model, self._model_path, self._model_name)

            else:
                self._log_value()

        else:
            if metric_val > self._curr_metric_val:
                print("\nSave_model metric_val " + str(metric_val) + " curr_metric_val " + str(self._curr_metric_val))
                self._curr_metric_val = metric_val
                save_model(self._model, self._model_path, self._model_name)
            else:
                self._log_value()


def save_model(model, segment_model_folder, segment_model_name):
    if not os.path.exists(segment_model_folder):
        os.mkdir(segment_model_folder)

    segment_model = tf.keras.Model(inputs=model.inputs, outputs=model.outputs)
    segment_model.save_weights(os.path.join(segment_model_folder, segment_model_name))


def load_weights(model, segment_model_folder, segment_model_name):
    if os.path.exists(os.path.join(segment_model_folder, f"{segment_model_name}.index")):
        print("Load")
        model.load_weights(os.path.join(segment_model_folder, segment_model_name))

        return model

    return 0
