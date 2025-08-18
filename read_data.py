
"""
Filename: read_data.py
Description: Used to load the property address and fire station address.
Author: Wei Zhang
Date: 7-17-2025

input: file name
output: address

Notice: the default path is in the Amazon S3

Property data folde path: s3://pr-home-datascience/Users/test/IntermediateDrive/Quantarium/AD/good_address/rundt=202504/

and Fire station folder: s3://pr-home-datascience/DSwarehouse/Datasources/FireStationLocation/FireStation/FireStations_0.csv


"""

import numpy as np
import pandas as pd
import s3fs
import boto3
from urllib.parse import urlparse

class ReadData:
    """Class for reading and handling CSV data."""
    def __init__(self, path, state='NT', directory=False):
        """
        Initialize the DataReader with a file path.

        Args:
            path (str): Path to the CSV file.
        """
        self.path = path
        self.data = None # Will read the data using the private function _read()
        self.state = state
        if not directory:
            self._read()
        if directory:
            self._read_directory()
        
    def _read(self):
        """Reads the CSV file into a DataFrame."""

        try:
            self.data = pd.read_csv(self.path, sep=',', dtype=str, low_memory=False)
            self.data = self.data[self.data['STATE'].isin([self.state])]
        except Exception as e:
            raise RuntimeError(f" Failed to read file: {e}")

    def _read_directory(self):
        """Reads the parquet file in certain directory into a DataFrame."""

        # Parse the S3 path
        parsed = urlparse(self.path)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip('/')


        # Initialize boto3 client and paginator
        s3 = boto3.client('s3')
        paginator = s3.get_paginator('list_objects_v2')
        
        # Find all .parquet files under the prefix
        all_files = []
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('.parquet'):
                        all_files.append(f"s3://{bucket}/{key}")

        if not all_files:
            print("No .parquet files found.")
            return pd.DataFrame()

        # Read and concatenate into one DataFrame
        self.data = pd.concat([pd.read_parquet(f) for f in all_files], ignore_index=True)
        

    def preview(self, n=5):
        """Print the first `n` rows of the data."""
        print(self.data.head(n))



if __name__ == "__main__":
    state = ['MA', 'CT']

    FS_path = "s3://pr-home-datascience/DSwarehouse/Datasources/FireStationLocation/FireStation/FireStations_0.csv"
    property_path = "s3://pr-home-datascience/Users/test/IntermediateDrive/Quantarium/AD/good_address/rundt=202504/" + 'state='+state[0] + '/'
    reader = ReadData(FS_path, state = state, directory=False)
    print(f'The property address is {property_path}')

    reader_property = ReadData(property_path, state=state[0], directory=True)
    print(reader.path)
    print(reader.preview())
    print(reader_property.preview())
    
    #temp = pd.read_csv(reader.path, sep = '\t', low_memory=False)
    #print(temp.head())