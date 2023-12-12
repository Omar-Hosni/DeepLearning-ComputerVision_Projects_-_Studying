import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data = {
    'age': [25, 30, 22, 35, 45],
    'income': [50000, 60000, 40000, 75000, 90000],
    'education': ['Bachelors', 'Masters', 'High School', 'PhD', 'Masters']
}
df = pd.DataFrame(data)

#1. Binning/Numerical Encoding
bins = [0, 25, 30, 35, np.inf]
labels = ['<25', '25-30', '30-40', '40+']
df['age_group'] = pd.cut(df['age'], bins, labels=labels)

#2. One-Hot Encoding
df = pd.get_dummies(df, columns=['education'], prefix='edu')

#3. Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['income_scaled'] = scaler.fit_transform(df['income'])

#4. Text Features - Extracting Length
df['edu_length'] = df['education'].apply(len)

#5. Date Features - Extracting Year and Month
df['year'] = pd.to_datetime('today').year
df['month'] = pd.to_datetime('today').month

#6. Interaction Features
df['age_income_interaction'] = df['age'] * df['income']

#7. Aggregation
education_grouped = df.groupby('education')['income'].mean()
education_grouped.rename(columns={'income':'avg_income_by_edu'}, inplace=True)
df = pd.merge(df, education_grouped, on='education', how='left')

#8. Time Since Event
df['days_since_last_purchase'] = (pd.to_datetime('today') - pd.to_datetime('2023-01-01')).dt.days

#9. Boolean Features
df['high_income'] = df['income'] > df['income'].mean()

#10. Feature Crosses
df['age_group_education'] = df['age_group'] + '_' + df['education']

print(df)


#Performing Bucketizing and Pruning

'''
Bucketizing and pruning are two techniques used in feature engineering and model optimization, respectively. Let's explore what each technique involves:

Bucketizing (Binning):
    Bucketizing, also known as binning, is a technique used to convert 
    continuous numerical features into categorical features by dividing the
     range of values into discrete intervals or "buckets." 
     This can help capture non-linear relationships between the feature and 
     the target variable. For example, 
     instead of using the exact age as a numerical feature, 
     you could divide it into age groups like "young," "middle-aged," 
     and "senior."

In the context of machine learning, bucketizing can be achieved
using various methods, such as the pd.cut() function in pandas or 
TensorFlow's tf.feature_column.bucketized_column. It's important to 
choose appropriate boundaries for the buckets to ensure the resulting
categorical feature captures meaningful information from the data.

Pruning:
    Pruning is a technique used in machine learning to optimize the 
    performance of complex models, particularly decision trees and neural
     networks. It involves removing parts of the model that are redundant
      or less important in order to prevent overfitting and improve 
      generalization to new, unseen data.

Pruning can refer to different techniques based on the type of model:

    Decision Trees: Decision tree pruning involves removing branches or nodes from a decision tree that do not contribute much to improving the accuracy of predictions on the test data. This helps simplify the tree, making it less prone to overfitting.
    Neural Networks: In the context of neural networks, pruning refers to removing specific neurons, connections, or layers from the network to reduce its complexity. This can be done using techniques like weight pruning (removing small-weight connections) or neuron pruning (removing less important neurons).
'''

age_bucketized = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('age')
)

age_educated_crossed = tf.feature_column.crossed_column(
    ['age_group',' education'],
    hash_bucket_size=1000
)

feature_columns = [
    age_bucketized,
    age_educated_crossed
]

model = keras.Sequential([
    layers.Input(shape=(len(feature_columns),)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

X = df[['age', 'age_group', 'education']]
y = df['income']

dataset = tf.data.Dataset.from_tensor_slices((dict(X),y))

train_size = int(0.8 * len(df))
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

model.fit(train_dataset.shuffle(buffer_size=train_size).batch(32), epochs=50)
test_loss = model.evaluate(test_dataset.batch(32))
print("Test Loss:", test_loss)