import tensorflow as tf
import numpy as np
import tensorflow_model_optimization as tfmot

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Load some example data
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 784)).astype(np.float32) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1)

# Perform post-training quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()

# Save the quantized model to a file
with open('quantized_model.tflite', 'wb') as f:
    f.write(tflite_quantized_model)


# Define a sparsity parameter for pruning
sparsity = 0.5

# Define a pruning schedule
pruning_schedule = tfmot.sparsity.keras.ConstantSparsity(sparsity, begin_step=0, end_step=1, frequency=1)

# Define a pruning callback
pruning_callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir='./logs')
]

# Prune the model
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

# Fine-tune the pruned model (optional)
pruned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
pruned_model.fit(x_train, y_train, epochs=1)

# Convert the pruned model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
tflite_pruned_model = converter.convert()

# Save the pruned model to a file
with open('pruned_model.tflite', 'wb') as f:
    f.write(tflite_pruned_model)