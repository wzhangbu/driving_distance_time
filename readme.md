



### Prerequisities
Env: conda create -n driving_distance python=3.12.11\
conda activate driving_distance\
<!---for ask_online.py--->
conda install -c conda-forge pandas numpy boto3  requests geopy ipython \
<!---for read_data.py--->
conda install -c conda-forge fsspec s3fs=2023.6.0 pyarrow  tqdm


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
