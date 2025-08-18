#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filename: ask_online.py
Description: Used to calculate the time and distance by requesting the OSRM website.
Author: Wei Zhang
Date: 7-16-2025

input: longitude and latitude (Array acceptable)
output: time, distance

Notice: the default path is in the Amazon S3

defaul path: s3://pr-home-datascience/Projects/AdHoc/InternProjects/2025/2025_Summer_AI_Property_Attributes/property_list_NJ_sample.csv
 

"""
import requests
import numpy as np
import pandas as pd
import time
import tqdm 
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
import tqdm.asyncio


def _process_batch_worker(args):
    '''
        This is for multi processors work in the Class
        args should include: 
            start_index for the loation of the batch
            origins: (lat, lon) for all origins data
            destinations: (lat, lon) for all destinations data
            batch size: how large the data is submitted to the table API 
            retry: if failed, how many times retry should be
            orsm_url: url for the orsm webstite
        returns:
            the panda df for the results. The default is origins[i] to destinations[i]
            time in minute, distance in miles
    '''

    start_idx, origins, destinations, batch_size, retry, orsm_url = args
    o_batch = origins[start_idx:start_idx + batch_size]
    d_batch = destinations[start_idx:start_idx + batch_size]
    n = len(o_batch)

    all_coords = o_batch + d_batch
    coord_str = ';'.join([f"{lon},{lat}" for lon, lat in all_coords])
    sources = ";".join(map(str, range(n)))
    dests = ";".join(map(str, range(n, 2 * n)))

    url = (
        f"{orsm_url}/table/v1/driving/{coord_str}"
        f"?sources={sources}&destinations={dests}&annotations=duration,distance"
    )

    for attempt in range(retry):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
            durations_matrix = np.array(data["durations"]) / 60.0  # to minutes
            distances_matrix = np.array(data["distances"]) / 1609.344  # to miles
            return (start_idx, [(durations_matrix[j][j], distances_matrix[j][j]) for j in range(n)])
        except Exception as e:
            print(f"Batch {start_idx}-{start_idx + n} attempt {attempt + 1} failed: {e}")

    return (start_idx, [(-1, -1)] * n)

class DrivingTimeDistanceOSRM:
    def __init__(self, max_threads=10, osrm_url='http://router.project-osrm.org/'):
        """
        Initialize the DrivingTimeDistanceOSRM with a threads number and the OSRM URL

        Args:
            max_threads (int): the number of threads used in the concurrent calculation
            orsm_ulr (str): link used to request the website route
        """
        self.orsm_url = osrm_url
        self.max_threads = max_threads
        
    def _route_one_pair(self, orig, dest):
        """
        calculate the time and distance for one route from the website

        Args:
            orig : tuple[float, float]
                Origin point as (longitude, latitude).
            dest : tuple[float, float]
                Destination point as (longitude, latitude).
        Returns:
            time_minutes : float
                Driving duration in minutes. Returns -1 if failed.
            distance_miles : float
                Driving distance in miles. Returns -1 if failed.

        response data structure is as follows:
        response = requests.request("GET", url, headers=headers, data=payload)

        response.text is {
            "destination_addresses" : [],
            "error_message" : "The provided API key is invalid. ",
            "origin_addresses" : [],
            "rows" : [],
            "status" : "REQUEST_DENIED"
            }

        """
        lon1, lat1 = orig
        lon2, lat2 = dest
        url = f"{self.orsm_url}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"

        try:
            # Ask online and request the answer
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            # print(data)
            route = data['routes'][0]
            duration_min = route['duration'] / 60 # seconds to minutes
            distance_miles = route['distance'] / 1609.344 # meter to miles

            return round(duration_min, 5), round(distance_miles, 5)
        except Exception as e:
            print(f"Failed for {orig} → {dest}: {e}")
            return -1, -1 

    def route_pairs_table_batched_conc(self, origins, destinations, batch_size=50, retry=3, max_threads=8):
        """
        calculate the time and distance for one route using OSRM/table 
        OSRM/table Computes the duration of the fastest route between all pairs of supplied coordinates.

        Args:
            orig : List of tuple[float, float]
                Origin point as (longitude, latitude).
            dest : List of tuple[float, float]
                Destination point as (longitude, latitude).
            batch_size: int
                Number of pairs per request
            retry: int
                number of reties per batch
            max_threads: int
                Number of concurrent threads
        Returns:
            time_minutes : List of float
                Driving duration in minutes. Returns -1 if failed.
            distance_miles : List of float
                Driving distance in miles. Returns -1 if failed.

        """
        assert len(origins) == len(destinations), 'The length of origins should be same as the destinations'
        durations_all = []
        distances_all = []
        total = len(origins)

        results = [None]*total

        def process_batch(start_idx):
            o_batch = origins[start_idx:start_idx + batch_size]
            d_batch = destinations[start_idx:start_idx + batch_size]
            n = len(o_batch)

            all_coords = o_batch + d_batch
            coord_str = ';'.join([f"{lon},{lat}" for lon, lat in all_coords])
            sources = ";".join(map(str, range(n)))
            dests = ";".join(map(str, range(n, 2 * n)))

            url = (
                f"{self.orsm_url}/table/v1/driving/{coord_str}"
                f"?sources={sources}&destinations={dests}&annotations=duration,distance"
            )

            for attempt in range(retry):
                try:
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    durations_matrix = np.array(data["durations"]) / 60.0  # to minutes
                    distances_matrix = np.array(data["distances"]) / 1609.344  # to miles
                    return (start_idx, [(durations_matrix[j][j], distances_matrix[j][j]) for j in range(n)])
                except Exception as e:
                    print(f"Batch {start_idx}-{start_idx + n} attempt {attempt + 1} failed: {e}")
            # all retries failed
            return (start_idx, [(-1, -1)] * n)

        futures = []
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            for i in tqdm.tqdm(range(0, total, batch_size), desc='Submitting Batches'):
            # for i in range(0, total, batch_size):
                futures.append(executor.submit(process_batch, i))
                time.sleep(0.5)

            for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc="Concurrent Batches"):
                start_idx, batch_result = future.result()
                results[start_idx:start_idx + len(batch_result)] = batch_result

        return results





















        '''
        assert len(origins) == len(destinations), 'The length of origins should be same as the destinations'
        durations_all = []
        distances_all = []
        total = len(origins)

        for i in tqdm.tqdm(range(0, total, batch_size), desc='Batches'):
            o_batch = origins[i:i+batch_size]
            d_batch = destinations[i:i+batch_size]
            n = len(o_batch)

            all_coords = o_batch + d_batch
            coord_str = ';'.join([f"{lon},{lat}" for lon, lat in all_coords])
            sources = ";".join(map(str, range(n)))
            dests = ";".join(map(str, range(n, 2 * n)))

            url = (
                f"{self.orsm_url}/table/v1/driving/{coord_str}"
                f"?sources={sources}&destinations={dests}&annotations=duration,distance"
            )

            # if requests fails, retry it several times
            for attempt in range(retry):
                try:
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    durations_matrix = np.array(data["durations"]) / 60.0 # seconds to minutes
                    distances_matrix = np.array(data["distances"]) / 1609.344 # meter to miles
                    durations_all.extend([durations_matrix[j][j] for j in range(n)])    #only recoord the diagonal
                    distances_all.extend([distances_matrix[j][j] for j in range(n)])    #only recoord the diagonal
                    break # success, breat the loop
                except Exception as e:
                    print(f"Batch {i}-{i+n} attempt {attempt+1} failed: {e}")
            else:
                durations_all.extend([-1] * n)
                distances_all.extend([-1] * n)

        return list(zip(durations_all, distances_all))
        '''

    async def _async_process_batch(self, session, origins, destinations, start_idx, batch_size, retry):
        """
        Async request for a single batch of origin-destination pairs

        Returns:
            Tuple: (start index, list of (duration_minutes, distance_miles))
        """
        o_batch = origins[start_idx:start_idx + batch_size]
        d_batch = destinations[start_idx:start_idx + batch_size]
        n = len(o_batch)

        all_coords = o_batch + d_batch
        coord_str = ';'.join([f"{lon},{lat}" for lon, lat in all_coords])
        sources = ";".join(map(str, range(n)))
        dests = ";".join(map(str, range(n, 2 * n)))

        url = (
            f"{self.orsm_url}/table/v1/driving/{coord_str}"
            f"?sources={sources}&destinations={dests}&annotations=duration,distance"
        )

        for attempt in range(retry):
            try:
                await asyncio.sleep(random.uniform(0.2, 1.0))
                async with session.get(url, timeout=45) as response:
                    response.raise_for_status()
                    data = await response.json()
                    durations_matrix = np.array(data["durations"]) / 60.0
                    distances_matrix = np.array(data["distances"]) / 1609.344
                    return (start_idx, [(durations_matrix[j][j], distances_matrix[j][j]) for j in range(n)])
            except Exception as e:
                print(f"[asyncio] Batch {start_idx}-{start_idx + n} attempt {attempt + 1} failed: {type(e).__name__}: {e}")
                await asyncio.sleep(1 + random.uniform(1, 2)) 

        return (start_idx, [(-1, -1)] * n)  # fallback if all retries fail

    async def route_pairs_table_async(self, origins, destinations, batch_size=50, retry=3):
        """
        Main coroutine: performs OSRM requests in async batches

        Returns:
            List of (duration_minutes, distance_miles) for each OD pair
        """
        assert len(origins) == len(destinations), 'Origins and destinations must be equal in length'
        total = len(origins)
        results = [None] * total

        connector = aiohttp.TCPConnector(limit=self.max_threads)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self._async_process_batch(session, origins, destinations, i, batch_size, retry)
                for i in range(0, total, batch_size)
            ]

            for future in tqdm.asyncio.tqdm.as_completed(tasks, total=len(tasks), desc="Async Batches"):
                start_idx, batch_result = await future
                results[start_idx:start_idx + len(batch_result)] = batch_result

        return results

    def route_pairs_table_batched_mp(self, origins, destinations, batch_size=50, retry=3, max_processes=8):
        """
        Multiprocessing version of route_pairs_table_batched_conc.
        """
        assert len(origins) == len(destinations), 'The length of origins should be same as the destinations'
        total = len(origins)
        results = [None] * total

        args_list = [
            (i, origins, destinations, batch_size, retry, self.orsm_url)
            for i in range(0, total, batch_size)
        ]

        with mp.get_context("spawn").Pool(processes=max_processes) as pool:
            for start_idx, batch_result in tqdm.tqdm(
                pool.imap_unordered(_process_batch_worker, args_list),
                total=len(args_list),
                desc="Multiprocessing Batches"
            ):
                results[start_idx:start_idx + len(batch_result)] = batch_result
                time.sleep(0.5)

        return results

    def compute(self, origins, destinations):
        '''
        This function is used to calculate estimated time and distance for arrays concurrently using _route_one_pair function
        
        Args:
            origins : list of tuple[float, float]
                Origin point as ( longitude, latitude) in a list
            destinations : list of  tuple[float, float]
                Destination point as ( longitude, latitude) in a list
        Returns:
            list of tuple[float, float]
                list of (time, distance), unit in minutes and miles
                returns (-1, -1 ) for failed queries

        '''
        if len(origins) != len(destinations):
            raise ValueError("origins and destinations must be of the same length")

        # create a list to store results, necessary for concurrent calculation
        results = [None] * len(origins)
        
        # Create a thread pool to perform concurrent requests
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {
                executor.submit(self._route_one_pair, orig, dest): i
                for i, (orig, dest) in enumerate(zip(origins, destinations))
            }
       
            # As each request finishes, store its result in the correct position
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results

if __name__ == "__main__":

    n = 1000
    origins = [(-74.019011, 40.219177)] * n
    destinations = [(-74.00082, 40.2166481)] * n

    random.seed(100) # fix the random seed to compare different methods
    origins = [(round(random.uniform(-74.02, -74.00), 4), 
                round(random.uniform(40.20, 40.23), 4)) for _ in range(n)]
    destinations = [(round(random.uniform(-74.02, -74.00), 4),
                round( random.uniform(40.20, 40.23), 4)) for _ in range(n)]

    url = 'http://router.project-osrm.org/'

    # assign the threads
    threads = 2


    router = DrivingTimeDistanceOSRM(max_threads=threads, osrm_url=url)

    start_time = time.time()

    # No table function, just multi-threads
    # results = router.compute(origins, destinations)

    # Use talble function and batch size
    # results = router.route_pairs_table_batched_conc(origins, destinations, batch_size=100, retry=2, max_threads=threads)
    results = asyncio.run(router.route_pairs_table_async(origins, destinations, batch_size=100, retry=2))
    # results = router.route_pairs_table_batched_mp(origins, destinations, batch_size=100, retry=1, max_processes=threads)

    print(f'The whole time is {time.time() - start_time}. seconds')
    

    
    for i, (time_hr, dist_miles) in enumerate(results):
        if i % int(n/10) == 0:
        # if i % 10 == 0 and i <= 110:
            print(f"Route {i+1}: {np.round(time_hr, 4)} min, {np.round(dist_miles, 5)} miles")

    