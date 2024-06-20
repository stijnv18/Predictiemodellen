import pandas as pd
import influxdb_client
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

token = "OqTOk2jgtV5Q7o4TCvcFJI2wzDqYaVpSbUgUv-FqjazIVRNAKyfdMZrmaLwn_0T-WXjSehFezqPEgzuIfZG0_w=="
org = "no"
url = "http://localhost:8086"

write_client = InfluxDBClient(url=url, token=token, org=org)
write_api = write_client.write_api(write_options=SYNCHRONOUS)

# Read the CSV file
df = pd.read_csv('customer_36v2.csv')

# Convert the timestamp column to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Write each row of the CSV file to the database
for index, row in df.iterrows():
    point = Point("Solar").time(time=row['timestamp'], write_precision=WritePrecision.NS)
    for column in df.columns:
        if column != 'timestamp':
            point.field(column, row[column])
    write_api.write(bucket="DataSet", org = "no", record=point)

write_api.__del__()
    
query_api = write_client.query_api()
query = """from(bucket: "DataSet")
 |> range(start: -15y)
 |> filter(fn: (r) => r._measurement == "Solar")"""
tables = query_api.query(query, org)

for table in tables:
    for record in table.records:
        print(record)