import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

# Load the ONNX model
onnx_model = onnx.load("your_model.onnx")

# Convert the ONNX model to a TensorFlow model
tf_rep = prepare(onnx_model)
tf_model = tf_rep.export_graph()
