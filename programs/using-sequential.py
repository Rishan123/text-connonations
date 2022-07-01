import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
# Schematically, the following Sequential model:
# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)

# A Sequential model is not appropriate when:
# Your model has multiple inputs or multiple outputs
# Any of your layers has multiple inputs or multiple outputs
# You need to do layer sharing
# You want non-linear topology (e.g. a residual connection, a multi-branch model)