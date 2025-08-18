
import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
import time
import os
import re
import glob as glob
import requests
import tqdm as tqdm
import s3fs


class ReadData:
    """Class for reading and handling CSV data."""
    def __init__(self, path, state='NT', directory=False):
        """
        Initialize the DataReader with a file path.

        Args:
            path (str): Path of the CSV file.
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


def latlon_to_xyz(lat, lon):
    '''
        convert latitduet and longitude to XYZ coordinate
        The lat and lon are in degrees by default

        The output is in XYZ coordinate system, where:
        - X is the cosine of latitude times the cosine of longitude
        - Y is the cosine of latitude times the sine of longitude  
        - Z is the sine of latitude
        XYZ is used for the kdtree 
    '''
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack((x, y, z))

def build_kdtree(firestation_coords):
    '''
        Build the kdtree use the firestation coordinates (in degree by default)
        return kdtree 
    '''
    xyz = latlon_to_xyz(
        np.array([lat for lat, lon in firestation_coords]),
        np.array([lon for lat, lon in firestation_coords])
    )
    return cKDTree(xyz), firestation_coords


def get_k_nearest_firestations(property_coords, kdtree, firestation_coords, k=5):
    '''
        Get the k nearest firestation using kdtree
        The propertiy coordinates are in (lat, lon) in degrees by default
        The firestations coordinates are in (lat, lon) in degrees by default
        kdtree is built using the XYZ coordinates of the firestations
        return the coordinates of the k nearest neighbors
    '''
    # conver the property address in XYZ
    xyz = latlon_to_xyz(
        np.array([lat for lat, lon in property_coords]),
        np.array([lon for lat, lon in property_coords])
    )
    # find the k nearest in kdtree
    dists, indices = kdtree.query(xyz, k=k)

    # record the Firestation candidates into [[address1, address2, address3, address4, address5]] for each property
    candidate_lists = [[firestation_coords[i] for i in row] for row in indices]
    return candidate_lists




def osrm_route_batch(properties, candidate_firestations, prop_id, fs_id, osrm_url="http://router.project-osrm.org/"):
    '''
        Use the kdtree and osrm_table to calculate the distance and time in batch size

        Args:
            properties : 
                Data frame of properties contains (longitude, latitude)
            candidate_firestations : List of tuple[float, float]
                Data frame of properties contains (longitude, latitude)
            prop_id: str
                qpid of the properties
            fs_id: str
                firestation id 
            osrm_url: str
                link that used to ask
        Returns:
            data frame contain the following information
                "qpid":,
                "property_lat",
                "property_lon",
                "fire_station_id",
                "station_lat",
                "station_lon",
                "travel_time_min",
                "travel_dist_mile":
            return [] if no travel route is found.
    '''

    # load all the firestation candidates and get rid of the repeated ones
    flat_candidates = [fs for group in candidate_firestations for fs in group]
    unique_fs = list({(lat, lon) for lat, lon in flat_candidates})
    fs_index = {fs: i for i, fs in enumerate(unique_fs)}

    src_strs = [f"{lon},{lat}" for lat, lon in properties]
    dst_strs = [f"{lon},{lat}" for lat, lon in unique_fs]

    all_coords = ";".join(src_strs + dst_strs)
    src_ids = list(range(len(properties)))
    dst_ids = list(range(len(properties), len(properties) + len(unique_fs)))

    # request online using osrm table API
    url = f"{osrm_url}/table/v1/driving/{all_coords}?sources={';'.join(map(str, src_ids))}&destinations={';'.join(map(str, dst_ids))}&annotations=duration,distance"

    # store the data into pd.dataframe
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        data = r.json()
        durations = data["durations"]
        distances = data["distances"]

        results = []
        for i in range(len(properties)):
            cand_idxs = [fs_index[fs] for fs in candidate_firestations[i]]
            best_idx = min(cand_idxs, key=lambda j: durations[i][j] if durations[i][j] is not None else float("inf"))
            best_dur = durations[i][best_idx]
            best_dist = distances[i][best_idx]
            best_fs = unique_fs[best_idx]

            results.append({
                "qpid": prop_id[i],
                "property_lat": properties[i][0],
                "property_lon": properties[i][1],
                "fire_station_id": fs_id[best_idx],
                "station_lat": best_fs[0],
                "station_lon": best_fs[1],
                "travel_time_min": best_dur / 60 if best_dur else None,
                "travel_dist_mile": best_dist / 1609.344 if best_dist else None
            })
        return results
    except Exception as e:
        print("OSRM error:", e)
        return []



def process_all_batches(properties_df, firestations_df, batchsize=100):
    '''
        Obtain the driving time and distance from the properties to the nearest firestations.
        Will use KDTree to get the potential 5 nearest fire stations.

        Args:
            properties_df : 
                Data frame of properties contains all variables, including (longitude, latitude)
            firestations_df : List of tuple[float, float]
                Data frame of properties contains all variables, including (longitude, latitude)
            batchsize:
                The batch size that will be submitted every job
                    
                            
        Returns:
            data frame contain the following information
                "qpid":,
                "property_lat",
                "property_lon",
                "fire_station_id",
                "station_lat",
                "station_lon",
                "travel_time_min",
                "travel_dist_mile":
            return [] if no travel route is found.

    '''
    # Load firestations and build KD-Tree
    # For firestation 'y' is latitude and 'x' is longitude
    firestations = list(zip(firestations_df['Y'], firestations_df['X']))
    fs_id = list(firestations_df['ID'])
    kdtree, firestations = build_kdtree(firestations)

    # 2. process the batch
    total = len(properties_df)
    results = []

    for start in tqdm(range(0, total, batchsize)):
        end = min(start + batchsize, total)
        batch = properties_df.iloc[start:end]
        # The properties are in (lat, lon) in degrees by default
        props = list(zip(batch['PA_Latitude'], batch['PA_Longitude']))
        prop_id = list(batch['QPID'])
        candidates = get_k_nearest_firestations(props, kdtree, firestations, k=5)

        batch_result = osrm_route_batch(props, candidates, prop_id, fs_id)
        results.extend(batch_result)

    return pd.DataFrame(results)



if __name__ == "__main__":
    states = ['MA']

    FS_path = "s3://pr-home-datascience/DSwarehouse/Datasources/FireStationLocation/FireStation/FireStations_0.csv"

    for state in states:
        output_path = f"./DRIVING_DISTANCE/results/"
        property_path = "s3://pr-home-datascience/Users/test/IntermediateDrive/Quantarium/AD/good_address/rundt=202504/" + 'state='+states[0] + '/'
        df_FS = ReadData(FS_path, state = state, directory=False)
        df_property = ReadData(property_path, state=state, directory=True)
    
        # results = run_large_property_batches(output_path, df_property.data, df_FS.data, state, batch_size=200000, process_batch_size=300)
        results = process_all_batches(output_path, df_property.data, df_FS.data, state, process_batch_size=300)
        