'''
we will use TFDV to generate a schema and record this process in the ML Metadata store.
You will be starting from scratch
so you will be defining each component of the data model.
The outline of steps involve:

    Defining the ML Metadata's storage database
    Setting up the necessary artifact types
    Setting up the execution types
    Generating an input artifact unit
    Generating an execution unit
    Registering an input event
    Running the TFDV component
    Generating an output artifact unit
    Registering an output event
    Updating the execution unit
    Seting up and generating a context unit
    Generating attributions and associations
'''

from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2
import urllib
import zipfile

import tensorflow as tf
import tensorflow_data_validation as tfdv

# Download the zip file from GCP and unzip it
url = 'https://storage.googleapis.com/artifacts.tfx-oss-public.appspot.com/datasets/chicago_data.zip'
zip, headers = urllib.request.urlretrieve(url)
zipfile.ZipFile(zip).extractall()
zipfile.ZipFile(zip).close()

connection_config = metadata_store_pb2.ConnectionConfig()
connection_config.fake_database.SetInParent()
store = metadata_store.MetadataStore(connection_config)

'''
Register ArtifactTypes

Next, you will create the artifact types needed and register them to the store. Since our simple exercise will just involve generating a schema using TFDV, you will only create two artifact types: one for the input dataset and another for the output schema. The main steps will be to:

    Declare an ArtifactType()
    Define the name of the artifact type
    Define the necessary properties within these artifact types. For example, it is important to know the data split name so you may want to have a split property for the artifact type that holds datasets.
    Use put_artifact_type() to register them to the metadata store. This generates an id that you can use later to refer to a particular artifact type.

'''

# Create ArtifactType for the input dataset
data_artifact_type = metadata_store_pb2.ArtifactType()
data_artifact_type.name = 'DataSet'
data_artifact_type.properties['name'] = metadata_store_pb2.STRING
data_artifact_type.properties['split'] = metadata_store_pb2.STRING
data_artifact_type.properties['version'] = metadata_store_pb2.INT

#Register artifact type to the Metadata Store
data_artifact_type_id = store.put_artifact_type(data_artifact_type)

#Create ArtifactType for Schema
schema_artifact_type = metadata_store_pb2.ArtifactType()
schema_artifact_type.name = 'Schema'
schema_artifact_type.properties['name'] = metadata_store_pb2.STRING
schema_artifact_type.properties['version'] = metadata_store_pb2.INT

#Register artifact type to the Metadata Store
schema_artifact_type_id = store.put_artifact_type(schema_artifact_type)

print('Data artifact type:\n', data_artifact_type)
print('Schema artifact type:\n', schema_artifact_type)
print('Data artifact type ID:', data_artifact_type_id)
print('Schema artifact type ID:', schema_artifact_type_id)

'''
Register ExecutionType

You will then create the execution types needed.
For the simple setup, you will just declare one 
for the data validation component with a state 
property so you can record if the process is running 
or already completed.
'''
# Create ExecutionType for Data Validation component
dv_execution_type = metadata_store_pb2.ExecutionType()
dv_execution_type.name = 'Data Validation'
dv_execution_type.properties['state'] = metadata_store_pb2.STRING

# Register execution type to the Metadata Store
dv_execution_type_id = store.put_execution_type(dv_execution_type)

print('Data validation execution type:\n', dv_execution_type)
print('Data validation execution type ID:', dv_execution_type_id)



'''
Generate input artifact unit

With the artifact types created,
you can now create instances of those types.
The cell below creates the artifact for the input dataset.
This artifact is recorded in the metadata store through the 
put_artifacts() function. Again, it generates an id
that can be used for reference.
'''

# Declare input artifact of type DataSet
data_artifact = metadata_store_pb2.Artifact()
data_artifact.uri = './data/train/data.csv'
data_artifact.type_id = data_artifact_type_id
data_artifact.properties['name'].string_value = 'Chicago Taxi Dataset'
data_artifact.properties['split'].string_value = 'train'
data_artifact.properties['version'].int_value = 1

#Submit input artifact to the Metadata Store
data_artifact_id = store.put_artifact_type([data_artifact])[0]

print('Data artifact:\n', data_artifact)
print('Data artifact ID:', data_artifact_id)

'''
Generate execution unit
Next, you will create an instance of the 
Data Validation execution type you registered earlier. 
You will set the state to RUNNING to signify that you are 
about to run the TFDV function. 
This is recorded with the put_executions() function.
'''
# Create ExecutionType for Data Validation component
dv_execution = metadata_store_pb2.ExecutionType()
dv_execution.type_id = dv_execution_type_id
dv_execution.properties['state'].string_value = 'RUNNING'

#Submit execution unit to the Metadata Store
dv_execution_id = store.put_executions([dv_execution])[0]

print('Data validation execution:\n', dv_execution)
print('Data validation execution ID:', dv_execution_id)

'''
Register input event
An event defines a relationship between artifacts and executions. 
You will generate the input event relationship for 
dataset artifact and data validation execution units. 
The list of event types are shown here 
and the event is recorded with the put_events() function.
'''

input_event = metadata_store_pb2.Event()
input_event.artifact_id = data_artifact_id
input_event.execution_id = dv_execution_id
input_event.type = metadata_store_pb2.Event.DECLARED_INPUT

store.put_events([input_event])
print('Input event:\n', input_event)

'''
Run the TFDV component
You will now run the TFDV component to generate the schema of dataset. 
This should look familiar since you've done this already in Week 1
'''
# Infer a schema by passing statistics to `infer_schema()`
train_data = './data/train/data.csv'
train_stats = tfdv.generate_statistics_from_csv(data_location=train_data)
schema = tfdv.infer_schema(statistics=train_stats)

schema_file = './schema.pbtxt'
tfdv.write_schema_text(schema, schema_file)

print("Dataset's Schema has been generated at:", schema_file)


'''
Generate output artifact unit

Now that the TFDV component has finished running and schema 
has been generated, you can create the artifact for the 
generated schema.
'''

# Declare output artifact of type Schema_artifact
schema_artifact = metadata_store_pb2.Artifact()
schema_artifact.uri = schema_file
schema_artifact.type_id = schema_artifact_type_id
schema_artifact.properties['version'].int_value = 1
schema_artifact.properties['name'].string_value = 'Chicago Taxi Schema'

# Submit output artifact to the Metadata Store
schema_artifact_id = store.put_artifacts([schema_artifact])[0]

print('Schema artifact:\n', schema_artifact)
print('Schema artifact ID:', schema_artifact_id)

'''
Register output event

Analogous to the input event earlier, 
you also want to define an output event to record 
the ouput artifact of a particular execution unit.

'''
# Declare the output event
output_event = metadata_store_pb2.Event()
output_event.artifact_id = schema_artifact_id
output_event.execution_id = dv_execution_id
output_event.type = metadata_store_pb2.Event.DECLARED_OUTPUT

# Submit output event to the Metadata Store
store.put_events([output_event])

print('Output event:\n', output_event)

'''
Update the execution unit

As the TFDV component has finished running successfully, 
you need to update the state of the execution unit and
record it again to the store.
'''

# Mark the `state` as `COMPLETED`
dv_execution.id = dv_execution_id
dv_execution.properties['state'].string_value = 'COMPLETED'

# Update execution unit in the Metadata Store
store.put_executions([dv_execution])

print('Data validation execution:\n', dv_execution)

'''
Setting up Context Types and Generating a Context Unit

You can group the artifacts and execution units into a Context. 
First, you need to define a ContextType which defines 
the required context. It follows a similar format as 
artifact and event types. You can register this with the 
put_context_type() function.
'''

#create context type
expt_context_type = metadata_store_pb2.ConextType()
expt_context_type.name=  'Experiment'
expt_context_type.properties['note'] = metadata_store_pb2.STRING

#register context type to the metadatastore
expt_context_type_id = store.put_context_type(expt_context_type)

#Similarly, you can create an instance of this context type and use the put_contexts() method to register to the store.

# Generate the context
expt_context = metadata_store_pb2.Context()
expt_context.type_id = expt_context_type_id
# Give the experiment a name
expt_context.name = 'Demo'
expt_context.properties['note'].string_value = 'Walkthrough of metadata'

# Submit context to the Metadata Store
expt_context_id = store.put_contexts([expt_context])[0]

print('Experiment Context type:\n', expt_context_type)
print('Experiment Context type ID: ', expt_context_type_id)

print('Experiment Context:\n', expt_context)
print('Experiment Context ID: ', expt_context_id)

'''
Generate attribution and association relationships

With the Context defined, you can now create its 
relationship with the artifact and executions you
previously used. You will create the relationship 
between schema artifact unit and experiment context 
unit to form an Attribution. Similarly, you will 
create the relationship between data validation 
execution unit and experiment context unit to form 
an Association. These are registered with the 
put_attributions_and_associations() method
'''

# Generate the attribution
expt_attribution = metadata_store_pb2.Attribution()
expt_attribution.artifact_id = schema_artifact_id
expt_attribution.context_id = expt_context_id

# Generate the association
expt_association = metadata_store_pb2.Association()
expt_association.execution_id = dv_execution_id
expt_association.context_id = expt_context_id

# Submit attribution and association to the Metadata Store
store.put_attributions_and_associations([expt_attribution], [expt_association])

print('Experiment Attribution:\n', expt_attribution)
print('Experiment Association:\n', expt_association)


'''
Retrieving Information from the Metadata Store¶

You've now recorded the needed information to the 
metadata store. If we did this in a persistent database,
 you can track which artifacts and events are related to 
 each other even without seeing the code used to generate 
 it. See a sample run below where you investigate what 
 dataset is used to generate the schema. (*It would be 
 obvious which dataset is used in our simple demo because 
 we only have two artifacts registered. Thus, assume that 
 you have thousands of entries in the metadata store.)

'''

#get artifacts types
store.get_artifact_types()
# Get 1st element in the list of `Schema` artifacts.
# You will investigate which dataset was used to generate it.
schema_to_inv = store.get_artifacts_by_type('Schema')[0]

# print output
print(schema_to_inv)

# Get events related to the schema id
schema_events = store.get_events_by_artifact_ids([schema_to_inv.id])

print(schema_events)

# Get events related to the output above
execution_events = store.get_events_by_execution_ids([schema_events[0].execution_id])

print(execution_events)

# Look up the artifact that is a declared input
artifact_input = execution_events[0]

store.get_artifacts_by_id([artifact_input.artifact_id])