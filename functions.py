'''
    function.py
    This file is to write the necessary codes for the property_FS.ipynb and complete_results.ipynb
    Ask_online.py has verified that the concurrent won't significantly accelerate the codes due to the IP block.
    Thus, we will only use osrm table API at one time.

    We will use the kdtree to do the nearest neighbor search
'''

import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
from tqdm import tqdm
import read_data
import os
import read_data
import functions
import re
import glob as glob
import requests

def latlon_to_xyz(lat, lon):
    '''
    Convert latitude and longitude to 3D Cartesian (XYZ) coordinates on a unit sphere.
    This format is used to build the KDTree for efficient nearest-neighbor search.
    
    Inputs:
        lat: array-like, latitude values in degrees
        lon: array-like, longitude values in degrees
    Returns:
        Nx3 numpy array of (x, y, z) coordinates
    '''

    # Convert input to NumPy arrays of type float64 for precision
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)

    # Convert degrees to radians for trigonometric functions
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Convert to 3D Cartesian coordinates on the unit sphere
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    # Stack the coordinates into an (N, 3) array
    return np.column_stack((x, y, z))


def build_kdtree(firestation_coords):
    '''
    Build a KDTree from fire station coordinates using 3D unit-sphere representation.

    Inputs:
        firestation_coords: list of (latitude, longitude) tuples
    Returns:
        A scipy.spatial.cKDTree object built from the fire station coordinates
    '''
    # Separate latitude and longitude
    latitudes = [lat for lat, lon in firestation_coords]
    longitudes = [lon for lat, lon in firestation_coords]

    # Convert to XYZ coordinates on the unit sphere
    xyz_coords = latlon_to_xyz(latitudes, longitudes)

    # Build and return the KDTree
    return cKDTree(xyz_coords)


def get_k_nearest_firestations(property_coords, kdtree, firestation_coords, k=5):
    '''
        Get the k nearest firestation using kdtree
        The propertiy coords are in (lat, lon) in degrees by default
        return the coordinates of the k nearest neighbors

    Inputs:
        property_coords: list of (latitude, longitude) tuples for properties (in degrees)
        kdtree: a cKDTree built from fire station coordinates (in XYZ)
        firestation_coords: original (lat, lon) list of fire stations for lookup
        k: number of nearest fire stations to return (default = 5)

    Returns:
        candidate_lists: list of lists, where each sublist contains the (lat, lon) coordinates
                         of the k nearest fire stations to a given property
    '''
    # conver the property address in 3D Cartesian XYZ
    xyz = latlon_to_xyz(
        np.array([lat for lat, lon in property_coords]),
        np.array([lon for lat, lon in property_coords])
    )
    # find the k nearest fire stations in kdtree
    dists, indices = kdtree.query(xyz, k=k)

    # Use the indices to retrieve original (lat, lon) coordinates of the nearest fire stations
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
            osrm_url: int
                Number of concurrent threads
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

    # Convert coordinates into OSRM API format: "lon,lat"
    src_strs = [f"{lon},{lat}" for lat, lon in properties]
    dst_strs = [f"{lon},{lat}" for lat, lon in unique_fs]

    # Create source and destination index lists for the OSRM request
    all_coords = ";".join(src_strs + dst_strs)
    src_ids = list(range(len(properties)))
    dst_ids = list(range(len(properties), len(properties) + len(unique_fs)))

    # request online using osrm table API
    url = f"{osrm_url}/table/v1/driving/{all_coords}?sources={';'.join(map(str, src_ids))}&destinations={';'.join(map(str, dst_ids))}&annotations=duration,distance"

    # store the data into pd.dataframe
    try:
        # Send GET request to OSRM and parse response
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        data = r.json()
        durations = data["durations"]
        distances = data["distances"]

        results = []
        # For each property, find the best (closest by time) fire station among its candidates
        for i in range(len(properties)):
            # Get indices of this property's candidate fire stations
            cand_idxs = [fs_index[fs] for fs in candidate_firestations[i]]
            # Choose the candidate with the minimum duration (skip None)
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



def process_all_batches(properties_df, firestations_df, batchsize=300):
    '''
    Process all properties in batches to compute the driving time and distance
    to the nearest fire station using the OSRM Table API.

    Args:
        properties_df: pd.DataFrame
            DataFrame containing property information with columns 'PA_Latitude', 'PA_Longitude', and 'QPID'
        firestations_df: pd.DataFrame
            DataFrame containing fire station information with columns 'X' (longitude), 'Y' (latitude), and 'ID'
        batchsize: int
            Number of properties to process per batch (default = 300)

    Returns:
        pd.DataFrame containing the OSRM results for all properties in that chunk
    '''
    # Load firestations and build KD-Tree
    # For firestation 'y' is latitude and 'x' is longitude
    firestations = list(zip(firestations_df['Y'], firestations_df['X']))
    fs_id = list(firestations_df['ID'])
    # Build KDTree from fire station coordinates
    kdtree, firestations = build_kdtree(firestations)

    # 2. process the batch
    total = len(properties_df)
    results = []

    for start in tqdm(range(0, total, batchsize)):
        end = min(start + batchsize, total)
        chunk = properties_df.iloc[start:end]

        # Extract property coordinates and IDs
        props = list(zip(chunk['PA_Latitude'], chunk['PA_Longitude']))
        prop_id = list(chunk['QPID'])

        # Find 5 nearest fire stations for each property using KDTree
        candidates = get_k_nearest_firestations(props, kdtree, firestations, k=5)

        # Use OSRM to get driving time and distance to best fire station
        batch_result = osrm_route_batch(props, candidates, prop_id, fs_id)
        # Accumulate results
        results.extend(batch_result)

    return pd.DataFrame(results)


def complete_osrm_results(
    full_data,
    firestations_df, 
    partial_result_path: str,

    output_path: str = "osrm_results_complete.csv",
    batch_size: int = 100,
    osrm_url: str = "http://router.project-osrm.org"
):
    """
    Complete missing OSRM results by rerunning the failed batches.

    Parameters:
full_data: pd.DataFrame
            Full original dataset containing all properties (must include 'QPID', 'property_lat', 'property_lon').
        firestations_df: pd.DataFrame
            Fire station information used to build the KDTree.
        partial_result_path: str
            Path to CSV file with partially completed OSRM results (must include 'qpid', 'travel_time_min', etc.).
        output_path: str
            File path to save the final merged results. Default: 'osrm_results_complete.csv'.
        batch_size: int
            Number of records to process per batch. Default: 100.
        osrm_url: str
            URL for the OSRM server. Default uses the public OSRM instance.
    """
    print(f"Loading input data...")
    # Ensure consistent column names for merging
    df_all = full_data.rename(columns={"QPID": "qpid" })
    df_result = pd.read_csv(partial_result_path)

    # Debug: check and report NaNs in travel_time_min
    dups = df_result['qpid'][df_result['qpid'].duplicated()]
    print(f"NaN in df_result: {df_result['travel_time_min'].isna().sum()}")
    nan_indices = df_result[df_result['travel_time_min'].isna()].index
    print("Indices with NaN travel_time_min:", nan_indices.tolist())
    print("Fill the NaN with 0.0.")
    # The nan will be replaced with 0.0
    # The NaN comes from the property is too close to the origin, the lon, and lat are exact the same.
    df_result['travel_time_min'] = df_result['travel_time_min'].fillna(0.0)
    df_result['travel_dist_mile'] = df_result['travel_dist_mile'].fillna(0.0)

    # Add row index to preserve original order after merging
    df_all["__row_index__"] = df_all.reset_index().index  # preserve order
    # Merge full dataset with partial results to identify missing entries
    df_merged = pd.merge(df_all, df_result, on=["qpid"] , how="left")

    # Filter rows that still have missing OSRM results
    df_missing = df_merged[df_merged["travel_time_min"].isna()].reset_index(drop=True)
    print(f"[!] Missing OSRM results for {len(df_missing)} rows. Starting retry...")

    results = []

    # for i in range(0, len(df_missing), batch_size):
    df_missing=df_missing.rename(columns={"qpid": "QPID"})

    # Retry processing only the missing entries
    df_retry = process_all_batches(df_missing, firestations_df, batchsize=batch_size)

    # NaN means the property is close to the destination, mark them 0.0
    df_retry['travel_time_min'] = df_retry['travel_time_min'].fillna(0.0)
    df_retry['travel_dist_mile'] = df_retry['travel_dist_mile'].fillna(0.0)

    # Ensure consistent string type for merging
    df_all["qpid"] = df_all["qpid"].astype(str)
    df_result["qpid"] = df_result["qpid"].astype(str)
    df_retry["qpid"] = df_retry["qpid"].astype(str)
    

    # # Merge the original partial results with the retried results
    df_final = pd.concat([df_result, df_retry], ignore_index=True)
    # Drop duplicates: keep first occurrence (original partial result), skip retry if exists
    df_final = df_final.drop_duplicates(subset="qpid", keep="first")

    # Merge will the original dataframe to get the correct row index
    df_final = pd.merge(df_final, df_all[["qpid", "__row_index__"]], on="qpid", how="left")
    print(f"After merging the final data has length of {len(df_final)}.")

    # to sort the df_final have the same order as original file
    df_final = df_final.sort_values("__row_index__")
    df_final = df_final.drop(columns="__row_index__").reset_index(drop=True)

    if len(df_all) == len(df_final):
        df_final.to_csv(output_path, index=False)
        print(f"[✓] Done! Final results with {len(df_final)} rows saved to {output_path}.")
    else:
        df_final.to_csv(output_path+'failed.csv', index=False)
        print("⚠️ Some results are missing or contain NaNs., not saved")

    


if __name__ == "__main__":
    states = ['MA']

    FS_path = "s3://pr-home-datascience/DSwarehouse/Datasources/FireStationLocation/FireStation/FireStations_0.csv"

    for state in states:
        output_path = f"./DRIVING_DISTANCE/results/"
        property_path = "s3://pr-home-datascience/Users/test/IntermediateDrive/Quantarium/AD/good_address/rundt=202504/" + 'state='+states[0] + '/'
        df_FS = read_data.ReadData(FS_path, state = state, directory=False)
        df_property = read_data.ReadData(property_path, state=state, directory=True)
    
        # results = run_large_property_batches(output_path, df_property.data, df_FS.data, state, batch_size=200000, process_batch_size=300)
        # results = process_all_batches(output_path, df_property.data, df_FS.data, state, process_batch_size=300)
        
        state = 'MA'
        # the paths for missing data and for the results
        missing_data_path = os.path.join('.', 'missing_data', state)
        output_path = os.path.join('.', 'complete_data', state)
        chunksizes = {'MA': 100000, 'NJ': 200000, 'NY': 300000}

        # obtain all the data file, format: "FS_batch_?_n"+chunksize +'failed.csv'
        index = 2
        chunksize = chunksizes[state]
        rerun_files = missing_data_path + f'/FS_batch_{index}_n{chunksize}_failed.csv'
        # the length of total data
        total = len(df_property.data)

        start = index * chunksize
        end = min((index + 1) * chunksize, total)
        df_chunk = df_property.data.iloc[start:end]

        tmp = complete_osrm_results(
            df_chunk,
            df_FS.data,
            rerun_files,
            output_path=output_path + f"/FS_batch_{index}_n{chunksize}.csv",
            batch_size=100,
            osrm_url = "http://router.project-osrm.org"
        )


    