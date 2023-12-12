import tensorflow as tf
import tensorflow_datasets as tfds

# Use TensorFlow Datasets to load and preprocess your data
def create_your_dataset():
    dataset, info = tfds.load("your_dataset_name", split="train", with_info=True)
    dataset = dataset.map(preprocess_function)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
