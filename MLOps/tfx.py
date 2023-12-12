import tensorflow as tf

from tfx import v1 as tfx

from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from google.protobuf.json_format import MessageToDict

import os
import pprint
pp = pprint.PrettyPrinter()

_pipeline_name = 'my_tfx_pipeline'
_pipeline_root = './pipeline/'
_data_root = './data/census_data'
_data_filepath = os.path.join(_data_root, 'adult.data')

context = InteractiveContext(pipelines_name=_pipeline_name, pipeline_root=_pipeline_root)

'''
this will split the data into training and evaluation sets (2/3 and 1/3)

convert each data row into tf.train.Example format. this protocol buffer 
designed for tensorflow operations and is used by the TFX components

compress and save the data collection under the _pipeline_root directory
for the other components to access. These examples are stored in
TFRecord format. This optimizes read and write operations within 
Tensorflow especially if you have a large collection of data

Its constructor takes the path to your data source/directory. In our
case, this is the _data_root path. The component supports several data
sources such as CSV, tf.Record, and BigQuery. Since our data is a CSV
file. we will use CsvExmpleGen to ingest the data.
'''
example_gen = tfx.components.CsvExampleGen(input_base=_data_root)
context.run(example_gen)

artifact = example_gen.outputs['examples'].get()[0]
print(f'split names: {artifact.split_names}')
print(f'artifact uri: {artifact.uri}')

train_uri = os.path.join(artifact.uri, 'Split-train')

tfrecord_filenames = [os.path.join(train_uri, name) for name in os.listdir(train_uri)]
dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type='GZIP')

def get_records(dataset, num_records):
    records = []

    for tfrecord in dataset.take(num_records):
        serialized_example = tfrecord.numpy()
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        example_dict = (MessageToDict(example))
        records.append(example_dict)

    return records

sample_records = get_records(dataset, 3)
pp.print(sample_records)

#statistic gen
statistics_gen = tfx.components.StatisticGen(
    examples=example_gen.outputs['examples']
)

context.run(statistics_gen)
context.show(statistics_gen.outputs['statistics'])

'''
SchemaGen
this component also uses TFDV to generate a schema based on your 
data statistics. As you've learned previously, a schema defines 
the expected bounds, types, and properties of the features in 
your dataset
'''

schema_gen = tfx.components.SchemaGen(
    statistics=statistics_gen.outputs['statistics']
)
context.run(schema_gen)
context.show(schema_gen.outputs['schema'])

'''
ExampleValidator
this component detects anomalies in your data based on the generated
schema from the previous step. Like the previous two components, it also
uses TFDV under the hood.

ExampleValidator will take as input the statistics from StatisticsGen
and the schema from SchemaGen. By default, it compares the statistics
from the evaluation split to the schema from the training split
'''
example_validator = tfx.components.ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)
context.run(example_validator)
context.show(example_validator.outputs['anomalies'])

#Transform

_census_constants_module_file = 'census_constants.py'

CATEGORICAL_FEATURE_KEYS = [
    'education', 'marital-status', 'occupation', 'race', 'relationship', 'workclass', 'sex', 'native-country'
]
NUMERIC_FEATURE_KEYS = ['fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
BUCKET_FEATURE_KEYS = ['age']
FEATURE_BUCKET_COUNT = {'age': 4}
LABEL_KEY = 'label'

def transformed_name(key):
    return key + '_xf'

# Set the transform module filename
_census_transform_module_file = 'census_transform.py'

import tensorflow as tf
import tensorflow_transform as tft

import census_constants
_NUMERIC_FEATURE_KEYS = census_constants.NUMERIC_FEATURE_KEYS
_CATEGORICAL_FEATURE_KEYS = census_constants.CATEGORICAL_FEATURE_KEYS
_BUCKET_FEATURE_KEYS = census_constants.BUCKET_FEATURE_KEYS
_FEATURE_BUCKET_COUNT = census_constants.FEATURE_BUCKET_COUNT
_LABEL_KEY = census_constants.LABEL_KEY
_transformed_name = census_constants.transformed_name

def preprocessing_fn(inputs):
    outputs = {}

    #Scale these features to the range [0,1]
    for key in _NUMERIC_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.scale_to_0_1(inputs[key])

    for key in _BUCKET_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.bucketize(
            inputs[key], _FEATURE_BUCKET_COUNT[key])

        # Convert strings to indices in a vocabulary
    for key in _CATEGORICAL_FEATURE_KEYS:
        outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(inputs[key])

    outputs[_transformed_name(_LABEL_KEY)] = tft.compute_and_apply_vocabulary(inputs[_LABEL_KEY])
    return outputs

tf.get_logger().setLevel('ERROR')
transform = tfx.components.Transform(examples=example_gen.outputs['examples'], schema=schema_gen.outputs['examples'], module_file=os.path.abspath(_census_constants_module_file))
context.run(transform)

transform_graph_uri = transform.outputs['transform_graph'].get()[0].uri
os.listdir(transform_graph_uri)


train_uri = os.path.join(transform.outputs['transformed_exampled']).get()[0].uri,'Split-train')
tfrecord_filenames = [os.path.join(train_uri, name) for name in os.listdir(train_uri)]
transformed_dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type='GZIP')

sample_records_xf = get_records(transformed_dataset, 3)
pp.pprint(sample_records_xf)

