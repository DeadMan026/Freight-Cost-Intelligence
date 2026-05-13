import pandas as pd
import os
from sqlalchemy import create_engine
import time
import logging

# Ensure logs directory exists relative to project root
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/ingestion_db.log',
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"                                                  
)

# Database in the data directory
engine = create_engine('sqlite:///data/inventory.db')

def load_raw_data():
    '''load the CSVs as dataframe and ingest into db'''
    start = time.time()
    
    # Path to CSVs
    data_path = 'data/CSVs'
    
    if not os.path.exists(data_path):
        logging.error(f"Directory {data_path} not found.")
        print(f"Error: Directory {data_path} not found. Please ensure CSVs are in {data_path}")
        return

    for file in os.listdir(data_path):
        if file.endswith('.csv'):
            table_name = file[:-4]
            logging.info(f"ingesting {file} into table {table_name}")
            print(f"Ingesting {file}...")

            chunks = pd.read_csv(os.path.join(data_path, file), chunksize=10000)
            
            # Use 'replace' for the first chunk to clear existing data,
            # then 'append' for all subsequent chunks.
            first_chunk = True
            for chunk in chunks:
                if first_chunk:
                    chunk.to_sql(table_name, con=engine, if_exists='replace', index=False)
                    first_chunk = False
                else:
                    chunk.to_sql(table_name, con=engine, if_exists='append', index=False)
                    
    end = time.time()
    '''start - end = seconds'''
    total_time = (end - start)/60          
    logging.info("--------------- Ingesting Complete-----------------")
    logging.info(f"\nTotal Time taken : {total_time:.2f} minutes")
    print(f"Ingestion complete. Total time: {total_time:.2f} minutes")

if __name__ == '__main__':
    load_raw_data()
