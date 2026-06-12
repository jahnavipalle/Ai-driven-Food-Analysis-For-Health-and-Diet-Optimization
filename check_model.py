from tensorflow.keras.models import load_model
import tensorflow as tf
import h5py

print("Current TF version:", tf.__version__)

model = load_model("food.h5", compile=False)
print("Model loaded successfully")

with h5py.File("food.h5", "r") as f:
    if "keras_version" in f.attrs:
        print("Keras version used:", f.attrs["keras_version"])
    if "tensorflow_version" in f.attrs:
        print("TensorFlow version used:", f.attrs["tensorflow_version"])
