



### Prerequisities
Env: conda create -n driving_distance python=3.12.11\
conda activate driving_distance
<!---for ask_online.py--->
conda install -c conda-forge pandas numpy boto3  requests geopy ipython 
<!---for read_data.py--->
conda install -c conda-forge fsspec s3fs pyarrow  tqdm

### Core Project Steps
  1. First need to finish the following steps (ask_online.py):
    a. We need to reproduce the previous project results (have been proved, no shown in this note).
    b. The parallel (concurrent  or multiprocessing) query won't accelerate the asking process (see the Appendix 1 for details).
    c. The table API gives 165 times faster query dealing with 100 data points. However, the concurrent of table does not accelerate with a larger dataset  (see the Appendix 2).
    d. Verified that Asyncio, multiprocessing, and concurrent won't accelerate the job with 1000 data, let alone 18 MM dataset (see the Appendix 3) .
  2. Next need to submit 4 jobs through Pipelines, which can run the request at the same time even using the same IP address. 4 jobs are a safe choice bashed on the experience, and 5 or more can easily lead to "Too many request" from the OSRM website.
  3. The configuration is in the Jupiter notebook property_FS.ipynb. 
  4. For the large datasets, we will divide them into different chunks with selected sizes, and batch size is 300 data points. For each property, the top 5 nearest fire stations are recorded and then obtain the unique stations for all properties. The request will contain (300, unique fire stations for the properties)and give the best route for each property. After finishing the requests for the chunk, save the data in csv file.
  5. There are times when request fails to give an answer (414 Client Error: Request-URI Too Large for url:……), please use complete_results.ipynb to obtain the missing properties.
  6. Use complete_results.ipynb to verify the data and merge them all into one single csv file for each state.


### File explainations
ask_online.py\
    This file is used to reproduce the previous project results. Verify the parallel query results and table API. Do not need it for the further data processing.

functions.py\
    This file includes the necessary functions used in the data processing. process_all_batches() and complete_osrm_results() will be used in the Jupiter notebooks.

read_data.py
    This file names the Class to read the data from Amazon S3.  

property_FS.ipynb\
    Use this Jupiter notebook to divide data into chunks, and request the job to the OpenStreetMap website with a batch size of 300. The results will be saved in S3 directory.

Complete_results.ipynb\
    This Jupiter notebook will complete the results which fail to request online with a 300 batch size, because it exceeds the URL limitation sometimes. Notebook will make the results in the same order of the original dataset, fill the NaN, and merge them into one CSV file. 

pipeline_jobs.ipynb and pipeline_jobs_complete.ipynb
    Use these files to submite jobs to Amazon Pipeline. Based on the experience, a safe choice is 4 concurrent jobs. Files are credited to Yi. pipeline_jobs.ipynb is for property_FS.ipynb and pipeline_jobs_complete.ipynb is for Complete_results.ipynb.


### Process 
  1. Divide the big dataset into different chunks, usually the chunksize is 300,000. Use 4 jobs in Pipelines to run the chunks, which request to OpenStreetMap with a batch size of 300. The 300 size is good for the most situation because the properties are close and 5 potential nearest firestations for each properties have repeated ones. The unique fire stations are usually dozens. It will return nothing if properties are disrete, which will give hundreds of potential fire stations and will be handled later. See property_FS.ipynb and pipeline_jobs.ipynb for details.
  
  2. For the results, we need to find out whether they miss some properties and whether they have NaN. We will use smaller batch size for the discrete properties, such as batchsize=100. We will fill with 0 for NaN because NaN for OpenStreetMap means two coordinates are too close to calculate. Save the final data and mark them as successful ones. See Complete_results.ipynb and pipeline_jobs_complete.ipynb for details.

  3. Examine the sequence of the successful outputs and check the order with the orgianl dataset. Merge then into one file if the order is the same. See Complete_results.ipynb for details.


### Notice
When using `asyncio` to query the [OSRM Table API](https://project-osrm.org/docs/v5.5.1/api/#table-service), it is important to control concurrency and request rate to avoid server-side rate limiting or blocking.


The OSRM server—especially the public demo server at [https://operations.osmfoundation.org/policies/api/](https://operations.osmfoundation.org/policies/api/)—is **not designed for high-volume production use**. Overloading it may result in:
- `429 Too Many Requests` responses
- Connection resets or timeouts
- Temporary IP bans

- **OSRM Table API Documentation**  
  https://project-osrm.org/docs/v5.5.1/api/#table-servic

- **Main asyncio module documentation**  
  https://docs.python.org/3/library/asyncio.html



  1. The OSRM website asks for (lon, lat).  lon is before lat.
  2. Download the maps from website: https://download.geofabrik.de/north-america/us-northeast.html, which should contain the states of interest (CT, MA, NH, NJ, NY, PA). The local OpenStreetMap with Docker on PC is a good option. Yet it requires the permission for the Docker installation and won't be used for this project.
  3. The table API on the OSRM website allows input of 10000 datapoints maximum, including the origins and destinations. The batch size 300 with unique nearby fire stations satisfies the requirement most of time.
  4. If the destination is too close to the origin, the table API may return empty, for example: from longitude latitude: (-71.0497036, 42.3666906 ) to (-71.0497036, 42.3666906 ): Route 1: 0.0 min, 0.0 miles. The table API return empty.
  5. For each properties, obtain the 5 nearest fire stations, and record the unique ones for all the batch, which will be calculated at the same time using table API. Then write down the best route for every property.
  6. The chuck size is based on the property number in certain state, details in property_FS.ipynb.
  7. We requires 4 jobs in Pipelines to divide the data and run the code. All jobs are using the same IP address. More jobs could be functional, but not robust, which means easily blocked by the OSRM website. The jobs in Pipeline won't expire automatically and can run for 2 days by default. Please check pipeline_jobs.ipynb (from Yi).
For the most cases, the 5 nearest stations for 300 batch properties are dozens, because the properties are in a neighbor. But if the properties are not close, please reduce the batch size for OSRM website query.