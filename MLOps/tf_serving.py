import json
import os
import tempfile
from urllib import request

import tensorflow as tf
from sklearn.model_selection import train_test_split


data = tf.keras.datasets.mnist.load_data()

X_train, y_train, X_test, y_test = train_test_split(data, test_size=0.2, shuffle=True)

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.reshape[0], 28, 28, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3,
                           strides=2, activation='relu', name='Conv1'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='Softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5)

result_eval = model.evaluate(X_test, y_test, verbose=0)

for metric, value in zip(model.metrics_names, result_eval):
    print(metric + ': {:.3}'.format(value))

MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))

if os.path.isdir(export_path):
    print('\n Already saved a model, cleaning up\n')

model.save(export_path, save_format='tf')

print('\nexport_path = {}'.format(export_path))

#send an inference request
data = json.dumps({'signature_name':'serving_default',
                   'instances': X_test[0:3].tolist()
                   })
headers = {'content-type':'application/json'}

url = 'http://localhost:8501/v1/models/digits_model:predict'
json_response = request.post(url, data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
