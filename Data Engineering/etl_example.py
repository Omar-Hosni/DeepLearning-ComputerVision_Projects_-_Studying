import prefect
from prefect import task, Flow
import pandas as pd
import sqlite3

@task
def extract_data():
    # Replace 'sales_data.csv' with your actual CSV file
    return pd.read_csv('sales_data.csv')

# Task to transform data (calculate total sales for each product in a chunk)
@task
def transform_chunk(chunk):
    return chunk.groupby('Product')['Sales'].sum().reset_index()

# Task to aggregate the transformed chunks
@task
def aggregate_data(transformed_chunks):
    return pd.concat(transformed_chunks).groupby('Product')['Sales'].sum().reset_index()

@task
def load_data(data):
    conn = sqlite3.connect('sales.db')
    data.to_sql('sales_summary', conn, if_exists='replace', index=False)
    conn.close()

if __name__ == '__main__':
    with Flow("SalesETL") as flow:
        data = extract_data()

        chunk_size = 1000

        data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        transformed_chunks = []
        for chunk in data_chunks:
            transformed_chunk = transform_chunk(chunk)
            transformed_chunks.append(transformed_chunk)

        aggregated_data = aggregate_data(transformed_chunks)
        load_data(aggregated_data)

    executor = prefect.executors.LocalDaskExecutor()

    flow_state = flow.run(executor=executor)

    if flow_state.is_successful():
        print("Sales ETL pipeline executed successfully.")

