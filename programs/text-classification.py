import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

tfds.disable_progress_bar()

def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

# The IMDB large movie review dataset is a binary classification dataset—all the reviews have either a positive or negative sentiment.
dataset, info = tfds.load('imdb_reviews', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
# Initially this returns a dataset of (text, label pairs):
for example, label in train_dataset.take(1):
  print('text: ', example.numpy())
  print('label: ', label.numpy())

# Next shuffle the data for training and create batches of these (text, label) pairs:
BUFFER_SIZE = 10000
BATCH_SIZE = 64
for example, label in train_dataset.take(1):
  print('texts: ', example.numpy()[:3])
  print()
  print('labels: ', label.numpy()[:1])

# The raw text loaded by tfds needs to be processed before it can be used in a model. 
# The simplest way to process text for training is using the TextVectorization layer. 
# This layer has many capabilities, but this tutorial sticks to the default behavior.
Create the layer, and pass the dataset's text to the layer's .adapt method:
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

# The .adapt method sets the layer's vocabulary. Here are the first 20 tokens. 
# After the padding and unknown tokens they're sorted by frequency:
vocab = np.array(encoder.get_vocabulary())

# Once the vocabulary is set, the layer can encode text into indices. 
# The tensors of indices are 0-padded to the longest sequence in the batch 
# (unless you set a fixed output_sequence_length):
encoded_example = encoder(example)[:3].numpy()

# With the default settings, the process is not completely reversible. There are three main reasons for that:

# The default value for preprocessing.TextVectorization's standardize argument is "lower_and_strip_punctuation".
# The limited vocabulary size and lack of character-based fallback results in some unknown tokens.

for n in range(3):
  print("Original: ", example[n].numpy())
  print("Round-trip: ", " ".join(vocab[encoded_example[n]]))
  print()

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        # Use masking to handle the variable sequence lengths
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

print([layer.supports_masking for layer in model.layers])
sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
predictions = model.predict(np.array([sample_text]))
print(predictions[0])
# predict on a sample text with padding

padding = "the " * 2000
predictions = model.predict(np.array([sample_text, padding]))
print(predictions[0])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset,
                    validation_steps=30)
test_loss, test_acc = model.evaluate(test_dataset)
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.ylim(None, 1)
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
plt.ylim(0, None)
sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
predictions = model.predict(np.array([sample_text]))

# Keras recurrent layers have two available modes that are controlled by the return_sequences constructor argument:
# If False it returns only the last output for each input sequence (a 2D tensor of shape 
# (batch_size, output_features)). This is the default, used in the previous model.
# If True the full sequences of successive outputs for each timestep is returned (a 3D tensor of 
# shape (batch_size, timesteps, output_features)).