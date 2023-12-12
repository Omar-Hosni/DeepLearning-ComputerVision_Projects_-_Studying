import tensorflow as tf
import tensorflow_transform as tft
import transforflow_model_analysis as tfma

transform_out_dir = ''

# Step1: Export EvalSavedModel for TFMA
def get_serve_tf_examples_fn(model, tf_transform_output):
    tf_transform_output = tft.TFTransformOutput(transform_output_dir)

    signatures = {
        'serving_default': get_serve_tf_examples_fn(model, tf).get_concrete_function(
            tf.TensorSpec(...)
        )
    }

    model.save(serving_model_dir_path, save_format='tf', signatures=signatures)


# Step2: Create EvalConfig
slice_spec = [slicer.SingleSliceSpec(columns=['column_name']), ...]

metrics = [tf.keras.metrics.Accuracy(name='accuracy'),
           tfma.metrics.MeanPrediction(name='mean_prediction'), ...]

metrics_specs = tfma.metrics.specs_from_metrics(metrics)

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key=features.LABEL_KEY)],
    slicing_specs=slice_spec,
    metrics_specs=metrics_specs, ...
)

# Step3: Analyze Model
eval_model_dir = ''
result_path = ''

eval_shared_model = tfma.default_eval_shared_model(
    eval_saved_model_path=eval_model_dir,
    eval_config=eval_config
)

eval_result = tfma.run_model_analysis(eval_shared_model=eval_shared_model,
                                      output_path=result_path)

# Step4: visualizing metrics
tfma.viewer.render_slicing_metrics(result)

