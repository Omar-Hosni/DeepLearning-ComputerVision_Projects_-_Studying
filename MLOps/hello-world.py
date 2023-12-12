'''
    ingest data from a base directory with ExampleGen
    compute the statistics of the training data with StatisticsGen
    infer a schema with SchemaGen
    detect anomalies in the evaluation data with ExampleValidator
    preprocess the data into features suitable for model training with Transform
'''

import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils

import pprint
import tempfile

raw_data = [
      {'x': 1, 'y': 1, 's': 'hello'},
      {'x': 2, 'y': 2, 's': 'world'},
      {'x': 3, 'y': 3, 's': 'hello'}
  ]


#define the metadata
raw_data_metadata = dataset_metadata.DatasetMetadata(
    schema_utils.schema_from_feature_spec({
        'y':tf.io.FixedLenFeature([], tf.float32),
        'x': tf.io.FixedLenFeature([], tf.float32),
        's': tf.io.FixedLenFeature([], tf.float32),
    })
)


def preprocessing_fn(inputs):
    x = inputs['x']
    y = inputs['y']
    s = inputs['s']

    x_centered = x - tft.mean(x)
    y_normalized = tft.scale_to_0_1(y)
    s_integerized = tft.compute_and_apply_vocubalry(s)
    x_centered_times_y_normalized = (x_centered * y_normalized)

    return {
        'x_centered': x_centered,
        'y_normalized': y_normalized,
        's_integerized': s_integerized,
        'x_centered_times_y_normalized': x_centered_times_y_normalized
    }

#generate a constant graph with the required transformation
tf.get_logger().setLevel('ERROR')

with tft_beam.Context(temp_dir=tempfile.mktemp()):
    transformed_dataset, transform_fn = (
        (raw_data, raw_data_metadata) | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn)
    )

transformed_data, transformed_metadata = transformed_dataset

print('\nRaw data:\n{}\n'.format(pprint.pformat(raw_data)))
print('Transformed data:\n{}'.format(pprint.pformat(transformed_data)))