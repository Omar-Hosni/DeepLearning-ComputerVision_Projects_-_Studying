import tensorflow as tf
import tensorflow_data_validation as tfdv

from tfx import v1 as tfx

from tfx.types import standard_artifacts

from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from google.protobuf.json_format import MessageToDict
from tensorflow_metadata.proto.v0 import schema_pb2

import os
import pprint
pp = pprint.PrettyPrinter()


# location of the pipeline metadata store
_pipeline_root = './pipeline/'

# directory of the raw data files
_data_root = './data/census_data'

# path to the raw training data
_data_filepath = os.path.join(_data_root, 'adult.data')

#Create the interactive context
context = InteractiveContext(pipelines_root=_pipeline_root)

#Example Gen
example_gen = tfx.components.CsvExampleGen(input_base=_data_root)

#StatisticsGen
statistics_gen = tfx.components.StatisticsGen(
    example_gen=example_gen.outputs['example']
)

#SchemaGen
schema_gen = tfx.components.SchemaGen(
    statistics=statistics_gen.outputs['statistics']
)
context.run(schema_gen)
context.show(schema_gen.outputs['schema'])

#Curating the Schema
schema_uri = schema_gen.outputs['schema']._artifacts[0].uri
schema = tfdv.load_scehma_text(os.path.join(schema_uri),'schema.pbtxt')

# Restrict the range of the `age` feature
tfdv.set_domain(schema, 'age', schema_pb2.IntDomain(name='age', min=17, max=90))

# Display the modified schema. Notice the `Domain` column of `age`.
tfdv.display_schema(schema)

#Schema Environments
schema.default_environment.append('TRAINING')
schema.default_environment.append('SERVING')
tfdv.get_feature(schema, 'label').not_in_environment.append("SERVING")

#You can now freeze the curated schema and save to a local directory.

# Declare the path to the updated schema directory
_updated_schema_dir = f'{_pipeline_root}/updated_schema'

# Declare the path to the schema file
schema_file = os.path.join(_updated_schema_dir, 'schema.pbtxt')

# Save the curated schema to the said file
tfdv.write_schema_text(schema, schema_file)

#Import Schema Gen
user_schema_importer = tfx.components.ImportSchemaGen(
    schema_file=schema_file
)

context.run(user_schema_importer, enable_cacher=False)
context.show(user_schema_importer.outputs['schema'])

#ExampleValidator
example_validator = tfx.components.ExapmleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=user_schema_importer.outputs['schema']
)
context.run(example_validator)
context.show(example_validator.outputs['anomalies'])

'''
Practice with ML Metadata
At this point, you should now take some time exploring 
the contents of the metadata store saved 
by your component runs.
This will let you practice tracking artifacts and how 
they are related to each other.
This involves looking at artifacts, executions, and events. 
This skill will let you recover related artifacts even without 
seeing the code of the training run. 
All you need is access to the metadata store.

'''

# Import mlmd and utilities
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

# Get the connection config to connect to the context's metadata store
connection_config = context.metadata_connection_config

# Instantiate a MetadataStore instance with the connection config
store = mlmd.MetadataStore(connection_config)

# Get artifact types
artifact_types = store.get_artifact_types()

# Print the results
[artifact_type.name for artifact_type in artifact_types]

# Get artifact types
schema_list = store.get_artifacts_by_type('Schema')

[(f'schema uri: {schema.uri}', f'schema id:{schema.id}') for schema in schema_list]

# Get 1st instance of ExampleAnomalies
example_anomalies = store.get_artifacts_by_type('ExampleAnomalies')[0]

# Print the artifact id
print(f'Artifact id: {example_anomalies.id}')

# Get first event related to the ID
anomalies_id_event = store.get_events_by_artifact_ids([example_anomalies.id])[0]

# Print results
print(anomalies_id_event)

# Get execution ID
anomalies_execution_id = anomalies_id_event.execution_id

# Get events by the execution ID
events_execution = store.get_events_by_execution_ids([anomalies_execution_id])

# Print results
print(events_execution)

# Filter INPUT type events
inputs_to_exval = [event.artifact_id for event in events_execution
                       if event.type == metadata_store_pb2.Event.INPUT]

# Print results
print(inputs_to_exval)