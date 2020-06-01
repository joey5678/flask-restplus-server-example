import requests
import json

from simple_rest_client.api import API

from mock.object import IMAGE_BASE64

api = API(
    api_root_url='http://localhost:5000/api/v1/',
    params={},
    headers={},
    timeout=2,
    append_slash=False,
    json_encode_body=True
)

api.add_resource(resource_name='analysis')

print(api.analysis.actions)



body = {'source_id': 'test',
        'request_type': 10,
        'sub_type': 'test',
        'content_id':'testttttttest',
        'content_type': 20,
        'image': IMAGE_BASE64}

headers = {"content-type": "application/json"}

# response = api.analysis.create(body=body, params={}, headers=headers)
# print(response.url)

# body = {"instances": [1.0, 2.0, 5.0]}

data = json.dumps(body)

url = 'http://10.240.108.54:8501/v1/models/half_plus_two:predict'
url = 'http://localhost:5000/api/v1/analysis/'
json_response = requests.post(url, data=data, headers=headers)
print(json_response.txt)

# predictions = json.loads(json_response.text)['predictions']
# print(predictions)