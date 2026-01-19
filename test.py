import json

import requests

url = "http://localhost:8080/2015-03-31/functions/function/invocations"


with open("test_sensor_data.json", "r") as f:
    sensor_data = json.load(f)

result = requests.post(url, json=sensor_data).json()
print(result)
