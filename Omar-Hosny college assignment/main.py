import re
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def extract_first_number(input_string):
    if isinstance(input_string, str):
        match = re.search(r'\b(\d+)(?:-\d+)?\b', input_string)

        if match:
            return int(match.group(1))
        else:
            return -1
    else:
        return -1




df = pd.read_csv('survey_results_public.csv')


for column in df.columns:
    unique_cols = df[column].unique()
    print(f"\nUnique elements in {column}:\n{unique_cols}")


import matplotlib.pyplot as plt

# Example: Histogram of Developer Ages
plt.hist(df['Age'], bins=20, edgecolor='black')
plt.title('Distribution of Developer Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Example: Countplot of Programming Languages
plt.figure(figsize=(12, 6))
language_count = df['LanguageHaveWorkedWith'].str.split(';', expand=True).stack().value_counts()
language_count.plot(kind='bar', color='skyblue')
plt.title('Most Popular Programming Languages')
plt.xlabel('Programming Language')
plt.ylabel('Count')
plt.show()

# Data preprocessing (cleaning and imputating data)

threshold_of_removing_col = df.shape[1]

numeric_columns = df.select_dtypes(include=['number']).columns
non_numeric_columns = df.select_dtypes(exclude=['number']).columns

# Fill missing values in numeric columns with the median
df_updated = df
df_updated['Age'] = df_updated['Age'].apply(extract_first_number)
df_updated['Age'] = df_updated['Age'].fillna(df_updated['Age'].median())

df_updated[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())


# cleaning df deleting all rows that has most of its elements as NA and storing output in df_cleaned
for idx, row in df_updated.iterrows():
    counter = 0
    for col in df_updated.columns:
        if row[col] == 'NA':
            counter += 1

    if counter > (len(df_updated.columns) // 4):
        df_updated = df_updated.dropna(axis=1, subset=[idx])

df_cleaned = df_updated

# Step2: Implementing cluster analaysis

features_for_clustering = ['Age', 'YearsCode', 'ConvertedCompYearly']

df_cluster = df_updated

print('df cluster', df_cluster[features_for_clustering])


# Convert 'Less than 1 year' to 1 and 'More than 50 years' to 50 in the 'YearsCode' column to make it all integers so we can later standardize
for i in range(len(df_cluster['YearsCode'])):
    if df_cluster['YearsCode'].iloc[i] == 'Less than 1 year':
        df_cluster.loc[df_cluster.index[i], 'YearsCode'] = 1
    elif df_cluster['YearsCode'].iloc[i] == 'More than 50 years':
        df_cluster.loc[df_cluster.index[i], 'YearsCode'] = 50

#imputer is something to impute missing data in certain columns
imputer = SimpleImputer(strategy='mean')
df_cluster[features_for_clustering] = imputer.fit_transform(df_cluster[features_for_clustering])

X = df_cluster[features_for_clustering]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans_results = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    kmeans_results.append(kmeans.inertia_)

# Plot the Elbow method
plt.plot(range(1, 11), kmeans_results, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

#Step 3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


numerical_features = ['Age', 'YearsCode']
categorical_features = ['LearnCode', 'LearnCodeOnline', 'MainBranch', 'DevType', 'OfficeStackSyncHaveWorkedWith']

# Drop rows with missing values in the categorical columns
df_updated.dropna(subset=categorical_features, inplace=True)

# Building pipeline for data transformation and ML algorithm which we will use to input inside it our data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Now let's create our data
features = numerical_features + categorical_features
target = 'HighIncome'

# Create a new dataframe with relevant features and the new 'HighIncome' column
df_classification = df_updated[features].copy()

# Define the new 'HighIncome' column
df_classification['HighIncome'] = (df_cluster['ConvertedCompYearly'] >= 50000).astype(int)

# Drop rows with missing values in the new 'HighIncome' column and features
df_classification.dropna(subset=['HighIncome'] + features, inplace=True)

# Split the data into training and testing sets
X = df_classification[features]
y = df_classification['HighIncome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43, shuffle=True)

# Train the model using the pipeline
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print(classification_report(y_test, y_pred))