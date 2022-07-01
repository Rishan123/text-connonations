import tensorflow as tf
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context

print("TensorFlow version:", tf.__version__)
# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
# Split the dataset into two categories: (training_labels, training_images) and (testing_labels, testing_images)
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert the sample data from integers to floating-point numbers:
x_train, x_test = x_train / 255.0, x_test / 255.0
# Create the Sequential model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
# For each example, the model returns a vector of logits or log-odds scores, one for each class.
predictions = model(x_train[:1]).numpy()
# The tf.nn.softmax function converts these logits to probabilities for each class:
tf.nn.softmax(predictions).numpy()
# Define a loss function for training using losses.SparseCategoricalCrossentropy, 
# which takes a vector of logits and a True index and returns a scalar loss for each example.
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.
# This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.
print(loss_fn(y_train[:1], predictions).numpy())
# Before you start training, configure and compile the model using Keras Model.compile. 
# Set the optimizer class to adam, set the loss to the loss_fn function you defined earlier, and specify a 
# metric to be evaluated for the model by setting the metrics parameter to accuracy.
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
# Use the Model.fit method to adjust your model parameters and minimize the loss:
model.fit(x_train, y_train, epochs=5)


