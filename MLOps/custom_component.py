import os
from tfx import components
from tfx.utils import docker_utils
from tfx.types import artifact_utils
from tfx.dsl.components.base import base_executor
from tfx.types import component_spec, standard_artifacts, artifact_utils
from tfx.dsl.component.experimental import container_component, placeholders

class CustomComponentSpec(component_spec.ComponentSpec):
    inputs = {}
    outputs = {
        'output': component_spec.ChannelParameter(type=standard_artifacts)
    }
    parameters = {}

class CustomExecutor(base_executor.BaseExecutor):

    def execute(self, input_dict, output_dict, exec_properties):
        docker_container_image = exec_properties['docker_image']
        docker_run_command = exec_properties['docker_command']

        docker_utils.run_container(
            image_name = docker_container_image,
            command = docker_run_command,
            output_dir = output_dict['output'].uri
        )

        output_dict['output'].uri = artifact_utils.get_single_uri(output_dict['output'])


component = CustomComponentSpec()

# Create a custom component
custom_component = components.create_component(
    component_name='CustomComponent',
    component_version='1.0',
    spec_class=CustomComponentSpec,
    executor_class=CustomExecutor,
    input_dict={},
    output_dict={'output': component.outputs['output']},
    exec_properties={}
)

'''
docker build -t custom-component-image .
docker tag custom-component-image:latest gcr.io/<project_id>/custom-component-image:latest
docker push gcr.io/<project_id>/custom-component-image:latest
'''

#TFX

from tfx.orchestration import pipeline
from tfx.orchestration.local_dag_runner import LocalDagRunner
from tfx.proto import example_gen_pb2, bulk_inferrer_pb2

# Define other TFX components (ExampleGen, StatisticsGen, etc.) as needed.
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, \
    Transform, Trainer, Tuner, InfraValidator, BulkInferrer
from tfx.proto import example_gen_pb2

# Define a CSVExampleGen component to ingest data
example_gen = CsvExampleGen(
    input_base='/path/to/data_dir',
    input_config=example_gen_pb2.Input(splits=[
        example_gen_pb2.Input.Split(name='train', pattern='train/*.csv'),
        example_gen_pb2.Input.Split(name='eval', pattern='eval/*.csv')
    ])
)

# Define a StatisticsGen component to compute statistics
statistics_gen = StatisticsGen(
    examples=example_gen.outputs['examples']
)

# Define a SchemaGen component to generate schema
schema_gen = SchemaGen(
    statistics=statistics_gen.outputs['statistics']
)

# Define an ExampleValidator component to check for anomalies
example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=schema_gen.outputs['schema']
)

# Define a Transform component to preprocess the data
transform = Transform(
    examples=example_gen.outputs['examples'],
    schema=schema_gen.outputs['schema'],
    module_file='/path/to/preprocessing_module.py'
)

# Define a Trainer component to train a model
trainer = Trainer(
    module_file='/path/to/trainer_module.py',
    examples=transform.outputs['transformed_examples'],
    schema=schema_gen.outputs['schema'],
    train_args={'num_steps': 1000},
    eval_args={'num_steps': 500}
)

# Define a Tuner component for hyperparameter tuning (optional)
tuner = Tuner(
    module_file='/path/to/tuner_module.py',
    examples=transform.outputs['transformed_examples'],
    schema=schema_gen.outputs['schema'],
    train_args={'num_steps': 1000},
    eval_args={'num_steps': 500},
    tune_args={'num_parallel_trials': 3}
)

# Define an InfraValidator component to validate the serving infrastructure (optional)
infra_validator = InfraValidator(
    model=trainer.outputs['model'],
    examples=example_gen.outputs['examples']
)

# Define a BulkInferrer component to perform batch inference (optional)
bulk_inferrer = BulkInferrer(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    output_config=bulk_inferrer_pb2.OutputConfig(output_data_format=bulk_inferrer_pb2.FORMAT_CSV)
)



custom_component = CustomComponent(
    docker_image='gcr.io/<project_id>/custom-component-image',
    docker_command=['python','/path/to/custom_script.py']
)

# Define the TFX pipeline.
pipeline_name = 'my_custom_pipeline'
pipeline = pipeline.Pipeline(
    pipeline_name=pipeline_name,
    pipeline_root='/path/to/pipeline_root',
    components=[
        # Add other components here.
        custom_component,
    ],
    enable_cache=False  # Disable cache for local execution.
)

# Run the TFX pipeline locally.
LocalDagRunner().run(pipeline)
