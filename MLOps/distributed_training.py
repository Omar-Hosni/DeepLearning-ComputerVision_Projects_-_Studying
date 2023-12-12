import tensorflow as tf

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = create_your_model()
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Load and preprocess your data using TensorFlow Datasets or other data pipelines
dataset = create_your_dataset()

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss_value = loss_fn(labels, predictions)
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss_value

for epoch in range(num_epochs):
    for batch in dataset:
        inputs, labels = batch
        loss = strategy.run(train_step, args=(inputs, labels))
        mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        print(f"Epoch {epoch}, Loss: {mean_loss}")
