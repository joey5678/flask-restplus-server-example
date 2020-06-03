import requests
import json
import sys


def test_jwt():

    body = {"identifier":"1122333311", "req_source":"appid567ee11ee"}
    headers = {"content-type": "application/json"}
    data = json.dumps(body)
    url = 'http://10.240.108.54:5000/auth'
    json_response = requests.post(url, data=data, headers=headers)
    assert json_response.text is not None, "No response"
    # print(json_response.text)
    status_code = json.loads(json_response.text)['status_code']
    assert status_code < 400, "wrong response"
    token = json.loads(json_response.text)['access_token']
    
    print(token)
    # Authorization: JWT eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZGVudGl0eSI6MSwiaWF0IjoxNDQ0OTE3NjQwLCJuYmYiOjE0NDQ5MTc2NDAsImV4cCI6MTQ0NDkxNzk0MH0.KPmI6WSjRjlpzecPvs3q_T3cJQvAgJvaQAPtk1abC_E
    headers['Authorization'] = f" JWT {token}"
    headers = {'Authorization': f"JWT {token}"}
    url = 'http://10.240.108.54:5000/protected'
    resp = requests.get(url, headers=headers)
    print(f"-={resp.text}=-")

if __name__ == '__main__':
    test_jwt()